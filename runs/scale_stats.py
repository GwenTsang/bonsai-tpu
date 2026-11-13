# scale statistics over all ternary matrices (live PARAMS; no reload)
mats=[PARAMS['head']]+[L[k] for L in PARAMS['layers'] for k in ('qkvz','qkv','out','ffn_gu','ffn_down') if k in L]
mn=1e9;mx=0;zeros=0;neg=0;tot=0;sub=0
for m in mats:
    s=m['s'];a=jnp.abs(s)
    mn=min(mn,float(jnp.min(jnp.where(a>0,a,jnp.inf))));mx=max(mx,float(jnp.max(a)));zeros+=int(jnp.sum(a==0));neg+=int(jnp.sum(s<0));sub+=int(jnp.sum((a>0)&(a<6.1035e-5)));tot+=s.size
print('scales',tot,'min_nonzero',mn,'max',mx,'zeros',zeros,'negative',neg,'f16_subnormal',sub,'log2 range',np.log2(mn),np.log2(mx),flush=True)
