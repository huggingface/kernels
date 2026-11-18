import kernels
import pytest
import torch
import torch.nn.functional as F

bfl = kernels.get_kernel("kernels-test/bfl-torch", version=1)


def _inputs(shape):
    torch.manual_seed(0)
    return tuple(torch.randn(shape, device="mps", dtype=torch.bfloat16) for _ in range(3))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("shape", [(1, 64, 1, 128), (2, 128, 3, 128)])
def test_shape_and_dtype(shape):
    q, k, v = _inputs(shape)
    out = bfl.dense_attention(q, k, v)
    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16
    assert out.device.type == "mps"


@pytest.mark.kernels_ci
def test_rejects_unsupported_inputs():
    q, k, v = _inputs((1, 64, 1, 128))
    with pytest.raises(TypeError):
        bfl.dense_attention(q.float(), k, v)
    with pytest.raises(ValueError):
        bfl.dense_attention(*_inputs((1, 96, 1, 128)))
    with pytest.raises(ValueError):
        bfl.dense_attention(*_inputs((1, 64, 1, 64)))
    with pytest.raises(ValueError):
        bfl.dense_attention(q, k, v[..., :64])


@pytest.mark.kernels_ci
def test_torch_compile():
    q, k, v = _inputs((1, 64, 1, 128))
    compiled = torch.compile(lambda q, k, v: bfl.dense_attention(q, k, v) * 2.0, fullgraph=True)
    assert torch.equal(compiled(q, k, v), bfl.dense_attention(q, k, v) * 2.0)


# Not a kernels_ci test: the output does not match SDPA on an M3 Max.
@pytest.mark.parametrize("shape", [(1, 64, 1, 128), (2, 128, 3, 128)])
def test_matches_sdpa(shape):
    q, k, v = _inputs(shape)
    out = bfl.dense_attention(q, k, v).float()

    # Torch SDPA wants [B, H, L, D].
    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
    ref = F.scaled_dot_product_attention(qf, kf, vf).transpose(1, 2)

    rel_rms = (out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert rel_rms.item() < 1e-2
