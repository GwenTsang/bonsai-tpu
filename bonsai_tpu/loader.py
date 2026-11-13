"""Bounded tensor reads from the audited GGUF with canonical NumPy axis order."""
import json
from pathlib import Path
import numpy as np
from .quantization import split_blocks,dequantize
from .hadamard import embedding_inverse,sign_table


class GGUFWeights:
    def __init__(self, path, metadata_path):
        self.path=Path(path)
        self.info=json.loads(Path(metadata_path).read_text())
        self.metadata=self.info['metadata']
        self.tensors={t['name']:t for t in self.info['tensors']}
        self.signs=sign_table(self.metadata)
        self.folded=set(self.metadata['prism.hadamard.weight_names'])
        self.size=self.path.stat().st_size
        with self.path.open('rb') as f:
            if f.read(4)!=b'GGUF':raise ValueError('Not a GGUF checkpoint')
        if self.metadata['general.architecture']!='qwen35':raise ValueError('Unsupported architecture')

    def raw(self,name,rows=None):
        t=self.tensors[name];shape=t['shape'];kind=t['type']
        num=int(np.prod(shape));offset=self.info['data_offset']+t['offset']
        group,bs={0:(1,4),1:(1,2),30:(1,2),142:(128,34),143:(128,28)}[kind]
        if num%group:raise ValueError('Invalid tensor group alignment')
        if rows is None:
            size=num//group*bs
            if offset+size>self.size:raise ValueError('Truncated checkpoint')
            with self.path.open('rb') as f:f.seek(offset);return f.read(size)
        if len(shape)!=2:raise ValueError('Row selection requires a matrix')
        stride=shape[0]//group*bs
        if shape[0]%group:raise ValueError('Invalid row group alignment')
        chunks=[]
        with self.path.open('rb') as f:
            for row in rows:
                if row<0 or row>=shape[1]:raise IndexError(row)
                f.seek(offset+int(row)*stride);chunk=f.read(stride)
                if len(chunk)!=stride:raise ValueError('Truncated checkpoint row')
                chunks.append(chunk)
        return b''.join(chunks)

    def dense(self,name,rows=None):
        t=self.tensors[name];raw=self.raw(name,rows);kind=t['type']
        if kind==0:a=np.frombuffer(raw,'<f4').copy()
        elif kind==1:a=np.frombuffer(raw,'<f2').astype(np.float32)
        elif kind==30:a=(np.frombuffer(raw,'<u2').astype(np.uint32)<<16).view(np.float32)
        elif kind in (142,143):a=dequantize(raw,kind)
        else:raise ValueError(kind)
        shape=list(reversed(t['shape']))
        if rows is not None:shape[0]=len(rows)
        return a.reshape(shape)

    def embeddings(self,ids):
        x=self.dense('token_embd.weight',rows=ids)
        if 'token_embd.weight' in self.metadata['prism.hadamard.inverse_weight_names']:
            x=embedding_inverse(x,self.signs[x.shape[-1]])
        return x

    def layer(self,index):
        result={}
        prefix=f'blk.{index}.'
        for name,t in self.tensors.items():
            if not name.startswith(prefix):continue
            key=name[len(prefix):]
            if t['type'] in (142,143):
                n,k=reversed(t['shape'])
                p,s=split_blocks(self.raw(name),t['type'])
                result[key]={'payload':p.reshape(n,k//128,-1),'scales':s.reshape(n,k//128)}
                # Type and transform contracts are fixed by this audited release.
                if t['type']!=143 or name not in self.folded:raise ValueError('Unsupported matrix contract')
            else:result[key]=self.dense(name)
        return result
