"""Minimal local GGUF reader: metadata, tensor directory, and memory-mapped tensor bytes."""
import mmap
import struct
import numpy as np

_FMT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
# ggml type -> (elements per block, bytes per block)
BLOCKS = {0: (1, 4), 1: (1, 2), 30: (1, 2), 142: (128, 34), 143: (128, 28)}
F32, F16, BF16, PQ2_0, PTQ1_0 = 0, 1, 30, 142, 143


class GGUF:
    def __init__(self, path):
        self.path = str(path)
        self._file = open(self.path, 'rb')
        self.map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        d = self.map
        if d[:4] != b'GGUF':
            raise ValueError('Not a GGUF file')
        self._pos = 4
        _version, n_tensors, n_kv = self._read('IQQ')
        self.metadata = {}
        for _ in range(n_kv):
            key = self._string()
            self.metadata[key] = self._value(self._read('I'))
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._string()
            ndim = self._read('I')
            shape = [self._read('Q') for _ in range(ndim)]
            kind, offset = self._read('IQ')
            self.tensors[name] = dict(name=name, shape=shape, type=kind, offset=offset)
        align = self.metadata.get('general.alignment', 32)
        self.data_offset = (self._pos + align - 1) // align * align

    def _read(self, fmt):
        fmt = '<' + fmt
        out = struct.unpack_from(fmt, self.map, self._pos)
        self._pos += struct.calcsize(fmt)
        return out[0] if len(out) == 1 else out

    def _string(self):
        n = self._read('Q')
        v = self.map[self._pos:self._pos + n].decode('utf-8', errors='replace')
        self._pos += n
        return v

    def _value(self, t):
        if t == 8:
            return self._string()
        if t == 9:
            subtype, n = self._read('IQ')
            if subtype in _FMT:
                fmt = '<%d%s' % (n, _FMT[subtype])
                out = list(struct.unpack_from(fmt, self.map, self._pos))
                self._pos += struct.calcsize(fmt)
                return out
            return [self._value(subtype) for _ in range(n)]
        return self._read(_FMT[t])

    def raw(self, name):
        """Zero-copy uint8 view of a tensor's bytes."""
        t = self.tensors[name]
        group, bs = BLOCKS[t['type']]
        n = int(np.prod(t['shape']))
        if n % group:
            raise ValueError('Bad block alignment for ' + name)
        size = n // group * bs
        start = self.data_offset + t['offset']
        if start + size > len(self.map):
            raise ValueError('Truncated checkpoint')
        return np.frombuffer(self.map, np.uint8, size, start)

    def dense(self, name):
        """Small non-ternary tensors as float32 in canonical (reversed GGML) axis order."""
        t = self.tensors[name]
        raw = self.raw(name)
        kind = t['type']
        if kind == F32:
            a = raw.view('<f4').astype(np.float32)
        elif kind == F16:
            a = raw.view('<f2').astype(np.float32)
        elif kind == BF16:
            a = (raw.view('<u2').astype(np.uint32) << 16).view(np.float32)
        else:
            raise ValueError('dense() is for float tensors; %s has type %d' % (name, kind))
        return a.reshape(list(reversed(t['shape'])))

    def ternary_blocks(self, name):
        """PTQ1_0 matrix [K, N] (ggml order) -> payload uint8 [N, K/128, 26], scales f16 [N, K/128]."""
        t = self.tensors[name]
        if t['type'] != PTQ1_0:
            raise ValueError('Expected PTQ1_0 for ' + name)
        k, n = t['shape']
        blocks = self.raw(name).reshape(n, k // 128, 28)
        payload = blocks[..., :26]
        scales = np.ascontiguousarray(blocks[..., 26:]).view('<f2').reshape(n, k // 128)
        return payload, scales
