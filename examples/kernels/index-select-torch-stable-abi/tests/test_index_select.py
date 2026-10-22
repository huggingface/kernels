import kernels
import pytest
import torch

index_select_torch_stable_abi = kernels.get_kernel(
    "kernels-test/index-select-torch-stable-abi", version=1
)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("indices", [[2, 0, 2], []])
def test_index_select(device, dim, contiguous, indices):
    if torch.device(device).type != "cuda":
        pytest.skip("This example requires CUDA")

    input = torch.arange(24, device=device, dtype=torch.float32).reshape(4, 6)
    if not contiguous:
        input = input.t()
    index = torch.tensor(indices, device=device, dtype=torch.int64)

    actual = index_select_torch_stable_abi.index_select(input, dim, index)
    expected = torch.index_select(input, dim, index)
    torch.testing.assert_close(actual, expected)
