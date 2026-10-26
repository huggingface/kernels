from unittest.mock import Mock

import pytest

from kernels.hf_hub import _check_trust_remote_code


def test_trust_check_failure_chains_original_error(monkeypatch):
    original = OSError("Cannot check publisher")
    api = Mock()
    api.get_organization_overview.side_effect = original
    monkeypatch.setattr("kernels.hf_hub._get_hf_api", lambda: api)

    with pytest.raises(ValueError, match="could not verify publisher trust status") as exc_info:
        _check_trust_remote_code("kernels-community/relu", local_files_only=False, trust_remote_code=False)

    assert exc_info.value.__cause__ is original
