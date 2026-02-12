# __KERNEL_NAME__

A custom kernel for PyTorch.

## Installation

```bash
pip install __REPO_ID__
```

## Usage

```python
import torch
from __KERNEL_NAME_NORMALIZED__ import __KERNEL_NAME_NORMALIZED__

# Create input tensor
x = torch.randn(1024, 1024, device="cuda")

# Run kernel
result = __KERNEL_NAME_NORMALIZED__(x)
```

## Development

### Building

```bash
nix develop
nix run .#build-and-copy
```

### Testing

```bash
nix develop .#test
pytest tests/
```

### Test as a `kernels` user

```bash
uv run example.py
```

## License

Apache 2.0
