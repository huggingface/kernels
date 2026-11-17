from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from kernels import has_kernel
from kernels.hf_hub import _check_trust_remote_code


@pytest.fixture
def api(monkeypatch):
    api = Mock(spec=HfApi)
    monkeypatch.setattr("kernels.hf_hub._get_hf_api", lambda: api)
    return api


@pytest.fixture
def unauthorized_error():
    response = httpx.Response(
        401,
        request=httpx.Request("GET", "https://huggingface.co/api/organizations/kernels-community/overview"),
        headers={"X-Request-Id": "test-request"},
    )
    return HfHubHTTPError("User Access Token is expired", response=response)


@pytest.mark.parametrize("trust_remote_code", [False, ["other/kernel"]])
def test_trust_check_authentication_failure(api, unauthorized_error, trust_remote_code):
    api.get_organization_overview.side_effect = unauthorized_error

    with pytest.raises(HfHubHTTPError, match="authentication failed") as exc_info:
        _check_trust_remote_code("kernels-community/relu", local_files_only=False, trust_remote_code=trust_remote_code)

    assert exc_info.value is unauthorized_error
    assert exc_info.value.response.status_code == 401
    assert exc_info.value.request_id == "test-request"
    message = str(exc_info.value)
    assert "User Access Token is expired" in message
    assert "kernels-community/relu" in message
    assert "invalid or expired" in message
    assert "hf auth login" in message
    assert "HF_TOKEN" in message
    assert "trust_remote_code" not in message


def test_has_kernel_propagates_authentication_failure(monkeypatch, api, unauthorized_error):
    monkeypatch.delenv("LOCAL_KERNELS", raising=False)
    monkeypatch.delenv("UNSTABLE_KERNELS_ALL_LATEST", raising=False)
    monkeypatch.setattr("kernels.load.constants.HF_HUB_OFFLINE", False)
    monkeypatch.setattr("kernels.load._get_hf_api", lambda: api)
    api.get_organization_overview.side_effect = unauthorized_error

    with pytest.raises(HfHubHTTPError, match="hf auth login") as exc_info:
        has_kernel("kernels-community/relu", version=1)

    assert exc_info.value is unauthorized_error
    api.get_organization_overview.assert_called_once_with("kernels-community")
    api.list_repo_refs.assert_not_called()
    api.hf_hub_download.assert_not_called()


@pytest.mark.parametrize("status_code", [403, 404, 500])
def test_trust_check_other_http_failure_chains_original_error(api, status_code):
    response = httpx.Response(status_code, request=httpx.Request("GET", "https://huggingface.co"))
    original = HfHubHTTPError("Cannot check publisher", response=response)
    api.get_organization_overview.side_effect = original

    with pytest.raises(ValueError, match="could not verify publisher trust status") as exc_info:
        _check_trust_remote_code("kernels-community/relu", local_files_only=False, trust_remote_code=False)

    assert exc_info.value.__cause__ is original


def test_trust_check_failure_chains_original_error(api):
    original = OSError("Cannot check publisher")
    api.get_organization_overview.side_effect = original

    with pytest.raises(ValueError, match="could not verify publisher trust status") as exc_info:
        _check_trust_remote_code("kernels-community/relu", local_files_only=False, trust_remote_code=False)

    assert exc_info.value.__cause__ is original


@pytest.mark.parametrize("trusted", [False, True])
def test_trust_check_publisher_status(api, trusted):
    api.get_organization_overview.return_value = SimpleNamespace(trustedKernelPublisher=trusted)

    if trusted:
        _check_trust_remote_code("kernels-community/relu", local_files_only=False, trust_remote_code=False)
    else:
        with pytest.raises(ValueError, match="not from a trusted publisher"):
            _check_trust_remote_code("kernels-community/relu", local_files_only=False, trust_remote_code=False)


@pytest.mark.parametrize(
    "local_files_only,trust_remote_code",
    [(True, False), (False, True), (False, ["kernels-community/relu"])],
)
def test_trust_check_bypasses_publisher_lookup(api, unauthorized_error, local_files_only, trust_remote_code):
    api.get_organization_overview.side_effect = unauthorized_error

    _check_trust_remote_code(
        "kernels-community/relu", local_files_only=local_files_only, trust_remote_code=trust_remote_code
    )

    api.get_organization_overview.assert_not_called()
