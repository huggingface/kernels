"""ReLU written in the TIRx kernel DSL with a symbolic length."""

from tvm.script import tirx as T

THREADS = 256


@T.prim_func
def relu(A_ptr: T.handle, B_ptr: T.handle):
    n = T.int32()
    A = T.match_buffer(A_ptr, (n,), "float32")
    B = T.match_buffer(B_ptr, (n,), "float32")

    T.device_entry()
    bx = T.cta_id([T.ceildiv(n, THREADS)])
    tx = T.thread_id([THREADS])

    i = bx * THREADS + tx
    if i < n:
        B[i] = T.max(A[i], T.float32(0.0))


__tirx_kernels__ = [relu]
