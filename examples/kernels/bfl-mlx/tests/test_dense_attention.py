import kernels
import mlx.core as mx
import pytest

bfl = kernels.get_kernel("kernels-test/bfl-mlx", version=1)


def _inputs(shape):
    mx.random.seed(0)
    return tuple(mx.random.normal(shape).astype(mx.bfloat16) for _ in range(3))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("shape", [(1, 64, 1, 128), (2, 128, 3, 128)])
def test_shape_and_dtype(shape):
    q, k, v = _inputs(shape)
    out = bfl.dense_attention(q, k, v)
    mx.eval(out)
    assert out.shape == q.shape
    assert out.dtype == mx.bfloat16


@pytest.mark.kernels_ci
def test_rejects_unsupported_inputs():
    q, k, v = _inputs((1, 64, 1, 128))
    with pytest.raises(TypeError):
        bfl.dense_attention(q.astype(mx.float32), k, v)
    with pytest.raises(ValueError):
        bfl.dense_attention(*_inputs((1, 96, 1, 128)))
    with pytest.raises(ValueError):
        bfl.dense_attention(*_inputs((1, 64, 1, 64)))
    with pytest.raises(ValueError):
        bfl.dense_attention(q, k, v[:, :, :, :64])


# Not a kernels_ci test: the output does not match SDPA on an M3 Max.
@pytest.mark.parametrize("shape", [(1, 64, 1, 128), (2, 128, 3, 128)])
def test_matches_sdpa(shape):
    q, k, v = _inputs(shape)
    out = bfl.dense_attention(q, k, v).astype(mx.float32)

    # MLX SDPA wants [B, H, L, D].
    qf, kf, vf = (a.astype(mx.float32).transpose(0, 2, 1, 3) for a in (q, k, v))
    ref = mx.fast.scaled_dot_product_attention(qf, kf, vf, scale=128**-0.5).transpose(0, 2, 1, 3)

    rel_rms = mx.sqrt(mx.mean((out - ref) ** 2)) / mx.sqrt(mx.mean(ref**2))
    assert rel_rms.item() < 1e-2
