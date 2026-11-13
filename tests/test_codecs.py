import ctypes
from pathlib import Path
import numpy as np
import pytest
from bonsai_tpu.quantization import decode_codes, dequantize, encode_ptq1, dequantize_kquant, kquant_codes, KQUANT_BYTES
from bonsai_tpu.hadamard import fwht, activation_transform, embedding_inverse, grouped_gdn

LIB = Path('upstream/prism-llama/build/bin/libggml-base.so')

@pytest.fixture(scope='module')
def reference():
    assert LIB.exists(), 'Build the pinned Prism reference library first'
    lib = ctypes.CDLL(str(LIB.resolve()))
    # Initialize ggml's FP16 lookup tables before calling dequantizers directly.
    class Params(ctypes.Structure):
        _fields_=[('mem_size',ctypes.c_size_t),('mem_buffer',ctypes.c_void_p),('no_alloc',ctypes.c_bool)]
    lib.ggml_init.argtypes=[Params];lib.ggml_init.restype=ctypes.c_void_p
    ctx=lib.ggml_init(Params(1024*1024,None,True))
    lib.ggml_free.argtypes=[ctypes.c_void_p];lib.ggml_free(ctx)
    return lib

@pytest.mark.parametrize('kind,size,payload_start,payload_end,scale_start',[(143,28,0,26,26),(142,34,2,34,0)])
def test_all_byte_values_against_c(reference,kind,size,payload_start,payload_end,scale_start):
    # Each possible byte at every payload position, plus random mixed patterns.
    n=payload_end-payload_start
    blocks=np.zeros((256*n+1024,size),np.uint8)
    for j in range(n):
        blocks[j*256:(j+1)*256,payload_start+j]=np.arange(256,dtype=np.uint8)
    rng=np.random.default_rng(47)
    blocks[-1024:,payload_start:payload_end]=rng.integers(0,256,(1024,n),dtype=np.uint8)
    scales=rng.uniform(-4,4,len(blocks)).astype('<f2')
    scales[:7]=[0,-0.,1,-1,2**-24,65504,-65504]
    blocks[:,scale_start:scale_start+2]=scales.view(np.uint8).reshape(-1,2)
    expected=np.empty((len(blocks),128),np.float32)
    fn=getattr(reference,'dequantize_row_ptq1_0' if kind==143 else 'dequantize_row_pq2_0')
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64]
    fn(blocks.ctypes.data,expected.ctypes.data,expected.size)
    np.testing.assert_array_equal(dequantize(blocks.tobytes(),kind),expected)


def test_roundtrip_ptq1():
    rng=np.random.default_rng(15)
    codes=rng.integers(-1,2,(1000,128),dtype=np.int8)
    packed=encode_ptq1(codes,np.ones(1000))
    np.testing.assert_array_equal(dequantize(packed.tobytes(),143),codes)


def test_hadamard_dense_and_inverse():
    h=np.ones((1,1),np.float32)
    for _ in range(10):h=np.block([[h,h],[h,-h]])
    h/=32
    rng=np.random.default_rng(2)
    x=rng.normal(size=(2,2048)).astype(np.float32)
    s=rng.choice([-1,1],2048).astype(np.float32)
    y=activation_transform(x,s)
    expected=((x*s).reshape(-1,1024)@h).reshape(x.shape)
    np.testing.assert_allclose(y,expected,atol=2e-6,rtol=1e-5)
    np.testing.assert_allclose(embedding_inverse(y,s),x,atol=1e-6,rtol=1e-5)


def test_grouped_head_permutation():
    x=np.arange(6144).reshape(1,-1)
    y=grouped_gdn(x).reshape(16,3,128)
    for k in range(16):
        for r in range(3):
            np.testing.assert_array_equal(y[k,r],x[0,(r*16+k)*128:(r*16+k+1)*128])


@pytest.mark.parametrize('kind,name,scale_at', [(10, 'dequantize_row_q2_K', 80), (11, 'dequantize_row_q3_K', 108)])
def test_kquants_against_c(reference, kind, name, scale_at):
    """Q2_K / Q3_K superblocks: every byte value at every payload position plus random blocks, bit-exact."""
    size = KQUANT_BYTES[kind]
    blocks = np.zeros((256 * scale_at + 2048, size), np.uint8)
    for j in range(scale_at):
        blocks[j * 256:(j + 1) * 256, j] = np.arange(256, dtype=np.uint8)
    rng = np.random.default_rng(kind)
    blocks[-2048:, :scale_at] = rng.integers(0, 256, (2048, scale_at), dtype=np.uint8)
    scales = rng.uniform(-4, 4, (len(blocks), (size - scale_at) // 2)).astype('<f2')
    blocks[:, scale_at:] = scales.view(np.uint8).reshape(len(blocks), -1)
    expected = np.empty((len(blocks), 256), np.float32)
    fn = getattr(reference, name)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn(blocks.ctypes.data, expected.ctypes.data, expected.size)
    np.testing.assert_array_equal(dequantize_kquant(blocks.tobytes(), kind), expected)
    codes, _ = kquant_codes(blocks, kind)
    assert codes.min() >= -124 and codes.max() <= 128          # the range the P8 byte layout relies on
