"""Extract real GGUF rows and compare them with the pinned C decoder."""
import ctypes,json
from pathlib import Path
import numpy as np
from bonsai_tpu.quantization import dequantize,split_blocks
from bonsai_tpu.hadamard import sign_table,activation_transform
from tests.test_codecs import reference
# Invoke the same C-library fixture initialization outside pytest.
lib=reference.__wrapped__()
m=json.loads(Path('research/gguf-metadata.json').read_text())
model=Path('weights/Ternary-Bonsai-2-27B-PTQ1_0.gguf')
rng=np.random.default_rng(20261113)
rows=[];vectors={};signs=sign_table(m['metadata'])
with model.open('rb') as f:
    for name in ['token_embd.weight','output.weight','blk.0.attn_qkv.weight','blk.0.ffn_down.weight','blk.0.ssm_out.weight','blk.3.attn_q.weight']:
        t=next(t for t in m['tensors'] if t['name']==name)
        k,n=t['shape'];row_bytes=k//128*28
        # Nonadjacent rows sample different groups and scales across the tensor.
        ids=np.array([0,1,n//2,n-1])
        raw=[]
        for row in ids:
            f.seek(m['data_offset']+t['offset']+int(row)*row_bytes);raw.append(f.read(row_bytes))
        raw=b''.join(raw)
        got=dequantize(raw,143).reshape(4,k)
        expected=np.empty_like(got);buf=np.frombuffer(raw,np.uint8)
        fn=lib.dequantize_row_ptq1_0
        fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64]
        fn(buf.ctypes.data,expected.ctypes.data,expected.size)
        np.testing.assert_array_equal(got,expected)
        payload,scale=split_blocks(raw,143)
        prefix=name.replace('.','_')
        vectors[prefix+'_payload']=payload
        vectors[prefix+'_scales']=scale
        vectors[prefix+'_decoded']=expected
        vectors[prefix+'_signs']=signs[k]
        x=rng.normal(size=(2,k)).astype(np.float32)
        vectors[prefix+'_x']=x
        vectors[prefix+'_transformed']=activation_transform(x,signs[k],gdn='.ssm_out.' in name)
        rows.append(dict(name=name,shape=t['shape'],row_ids=ids.tolist(),max_error=0.0))
Path('results/real-weight-codecs.json').write_text(json.dumps(rows,indent=2))
np.savez('results/real-weight-vectors.npz',**vectors)
print(json.dumps(rows,indent=2))
