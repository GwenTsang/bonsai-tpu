"""Correctness-first Qwen35 layers; FP32 state, packed PTQ1 weights in HBM."""
import jax
import jax.numpy as jnp
from .quantization import decode_codes
from .hadamard import activation_transform


def rms_norm(x,weight,eps=1e-6):
    x=x.astype(jnp.float32)
    return x*jax.lax.rsqrt(jnp.mean(x*x,axis=-1,keepdims=True)+eps)*weight


def l2_norm(x,eps=1e-6):
    # Prism CPU source uses max(sqrt(sum(x*x)), eps), not sqrt(sum+eps).
    return x/jnp.maximum(jnp.sqrt(jnp.sum(x*x,axis=-1,keepdims=True)),eps)


def linear(x,w,signs=None,gdn=False):
    if isinstance(w,dict):
        x=activation_transform(x,signs,gdn=gdn,xp=jnp)
        p,s=w['payload'],w['scales']
        a=(decode_codes(p,143,jnp).astype(jnp.float32)*s[...,None]).reshape(p.shape[0],-1)
    else:a=w
    return jnp.matmul(x,a.T,precision=jax.lax.Precision.HIGHEST)


def causal_conv(x,kernel,state):
    """x[T,C], kernel[C,4], state[3,C] oldest first."""
    def step(state,xt):
        window=jnp.concatenate((state,xt[None,:]),axis=0)
        y=jnp.sum(window*kernel.T,axis=0)
        return window[1:],y
    state,y=jax.lax.scan(step,state,x)
    return y,state


def delta_net(q,k,v,alpha,beta,dt,a,state):
    """q/k[T,Hk,D], v[T,Hv,D], state[Hv,Dv,Dk] in tiled head order."""
    heads=jnp.arange(v.shape[1])%q.shape[1]
    q=q[:,heads,:];k=k[:,heads,:]
    beta=jax.nn.sigmoid(beta)
    gate=a*jax.nn.softplus(alpha+dt)
    def step(state,values):
        qt,kt,vt,gt,bt=values
        state=state*jnp.exp(gt)[:,None,None]
        prediction=jnp.sum(state*kt[:,None,:],axis=-1)
        delta=(vt-prediction)*bt[:,None]
        state=state+delta[:,:,None]*kt[:,None,:]
        out=jnp.sum(state*qt[:,None,:],axis=-1)*(q.shape[-1]**-0.5)
        return state,out
    state,out=jax.lax.scan(step,state,(q,k,v,gate,beta))
    return out,state


def feed_forward(x,w,signs):
    gate=linear(x,w['ffn_gate.weight'],signs[x.shape[-1]])
    up=linear(x,w['ffn_up.weight'],signs[x.shape[-1]])
    activated=jax.nn.silu(gate)*up
    out=linear(activated,w['ffn_down.weight'],signs[activated.shape[-1]])
    return out,dict(ffn_gate=gate,ffn_up=up,ffn_swiglu=activated,ffn_out=out)


def recurrent_block(x,w,signs,conv_state,state,eps=1e-6):
    trace={}
    def save(name,value):trace[name]=value;return value
    norm=save('attn_norm',rms_norm(x,w['attn_norm.weight'],eps))
    qkv=save('linear_attn_qkv_mixed',linear(norm,w['attn_qkv.weight'],signs[x.shape[-1]]))
    z=save('z',linear(norm,w['attn_gate.weight'],signs[x.shape[-1]]))
    alpha=save('alpha',linear(norm,w['ssm_alpha.weight']))
    beta=save('beta',linear(norm,w['ssm_beta.weight']))
    conv,conv_state=causal_conv(qkv,w['ssm_conv1d.weight'],conv_state)
    save('conv_output_raw',conv)
    conv=save('conv_output_silu',jax.nn.silu(conv))
    q=conv[:,:2048].reshape(-1,16,128);k=conv[:,2048:4096].reshape(-1,16,128)
    v=conv[:,4096:].reshape(-1,48,128)
    q=save('q_conv_predelta',l2_norm(q,eps));k=save('k_conv_predelta',l2_norm(k,eps))
    save('v_conv_predelta',v)
    out,state=delta_net(q,k,v,alpha,beta,w['ssm_dt.bias'],w['ssm_a'],state)
    save('attn_output',out);save('new_state',state)
    gated=rms_norm(out,w['ssm_norm.weight'],eps)*jax.nn.silu(z.reshape(out.shape))
    out=save('final_output',gated.reshape(x.shape[0],-1))
    out=save('linear_attn_out',linear(out,w['ssm_out.weight'],signs[out.shape[-1]],gdn=True))
    residual=save('attn_residual',x+out)
    norm=save('attn_post_norm',rms_norm(residual,w['post_attention_norm.weight'],eps))
    ff,fftrace=feed_forward(norm,w,signs);trace.update(fftrace)
    out=save('l_out',residual+ff)
    return out,conv_state,state,trace


def rope_text(x,positions,rotary_dim=64,base=10000000.0):
    """Text-only MRoPE: all positional axes coincide; rotate the first 64 dims."""
    half=rotary_dim//2
    freq=base**(-jnp.arange(half,dtype=jnp.float32)/half)
    theta=positions[:,None,None]*freq[None,None,:]
    c,s=jnp.cos(theta),jnp.sin(theta)
    first,second=x[...,:half],x[...,half:rotary_dim]
    return jnp.concatenate((first*c-second*s,second*c+first*s,x[...,rotary_dim:]),axis=-1)


def full_attention_block(x,w,signs,k_cache,v_cache,eps=1e-6):
    """Correctness path with growing FP16 caches; fixed/paged caches come later."""
    trace={}
    def save(name,value):trace[name]=value;return value
    norm=save('attn_norm',rms_norm(x,w['attn_norm.weight'],eps))
    qg=save('Qcur_full',linear(norm,w['attn_q.weight'],signs[x.shape[-1]]))
    qg=qg.reshape(-1,24,512)
    q=save('Qcur_reshaped',qg[...,:256])
    gate=save('gate_reshaped',qg[...,256:].reshape(-1,6144))
    q=save('Qcur_normed',rms_norm(q,w['attn_q_norm.weight'],eps))
    k=linear(norm,w['attn_k.weight'],signs[x.shape[-1]]).reshape(-1,4,256)
    v=save('Vcur',linear(norm,w['attn_v.weight'],signs[x.shape[-1]]).reshape(-1,4,256))
    k=save('Kcur_normed',rms_norm(k,w['attn_k_norm.weight'],eps))
    start=k_cache.shape[0];positions=start+jnp.arange(x.shape[0])
    q=save('Qcur',rope_text(q,positions));k=save('Kcur',rope_text(k,positions))
    k_cache=jnp.concatenate((k_cache,k.astype(k_cache.dtype)),axis=0)
    v_cache=jnp.concatenate((v_cache,v.astype(v_cache.dtype)),axis=0)
    head_ids=jnp.arange(24)//6
    keys=k_cache[:,head_ids,:].astype(jnp.float32);values=v_cache[:,head_ids,:].astype(jnp.float32)
    scores=jnp.einsum('thd,shd->hts',q,keys,precision=jax.lax.Precision.HIGHEST)*(256**-0.5)
    mask=jnp.arange(keys.shape[0])[None,:]<=positions[:,None]
    probs=jax.nn.softmax(jnp.where(mask[None,...],scores,-jnp.inf),axis=-1)
    out=jnp.einsum('hts,shd->thd',probs,values,precision=jax.lax.Precision.HIGHEST).reshape(-1,6144)
    save('attn_pregate',out)
    out=save('attn_gated',out*save('gate_sigmoid',jax.nn.sigmoid(gate)))
    out=save('attn_output',linear(out,w['attn_output.weight'],signs[out.shape[-1]]))
    residual=save('attn_residual',x+out)
    norm=save('attn_post_norm',rms_norm(residual,w['post_attention_norm.weight'],eps))
    ff,fftrace=feed_forward(norm,w,signs);trace.update(fftrace)
    out=save('l_out',residual+ff)
    return out,k_cache,v_cache,trace
