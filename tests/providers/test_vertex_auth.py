"""Shared Vertex AI authentication tests (issue #249).

The three Vertex providers (language, embedding, TTS) share one auth path via
VertexAuthMixin. These verify that a configured service-account key file is
actually used — the bug was that TTS/embedding ignored it and shelled out to the
gcloud CLI. Everything is mocked; no real credentials or network.
"""

from unittest.mock import MagicMock, patch

import pytest

from esperanto.providers.vertex_auth import VertexAuthMixin


class _Dummy(VertexAuthMixin):
    def __init__(self, config=None, credentials_file=None):
        self._config = config or {}
        if credentials_file is not None:
            self.credentials_file = credentials_file


@pytest.fixture
def fake_creds():
    creds = MagicMock()
    creds.valid = True
    creds.token = "sa-token"
    return creds


# --- credential resolution ------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"config": {"credentials_path": "/fake/key.json"}},  # downstream alias
        {"config": {"credentials_file": "/fake/key.json"}},  # config key
        {"credentials_file": "/fake/key.json"},              # attribute (llm field)
    ],
    ids=["credentials_path", "credentials_file_config", "credentials_file_attr"],
)
def test_service_account_file_is_used(kwargs, fake_creds):
    with patch(
        "google.oauth2.service_account.Credentials.from_service_account_file",
        return_value=fake_creds,
    ) as mock_load:
        d = _Dummy(**kwargs)
        d._load_credentials()
    mock_load.assert_called_once()
    assert mock_load.call_args[0][0] == "/fake/key.json"
    assert d._credentials is fake_creds


def test_env_var_routes_through_adc_not_service_account(monkeypatch, fake_creds):
    """GOOGLE_APPLICATION_CREDENTIALS is left for ADC (google.auth.default) to
    interpret — never forced through from_service_account_file, which would break
    workload/workforce identity federation configs."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/env/federated.json")
    with patch(
        "google.oauth2.service_account.Credentials.from_service_account_file"
    ) as mock_sa, patch(
        "google.auth.default", return_value=(fake_creds, "proj")
    ) as mock_adc:
        d = _Dummy()
        d._load_credentials()
    mock_sa.assert_not_called()
    mock_adc.assert_called_once()
    assert d._credentials is fake_creds


# --- token acquisition ----------------------------------------------------- #


def test_token_uses_credentials_not_gcloud(fake_creds):
    d = _Dummy()
    d._credentials = fake_creds
    with patch("esperanto.providers.vertex_auth.subprocess.run") as mock_run:
        token = d._get_access_token()
    assert token == "sa-token"
    mock_run.assert_not_called()  # the whole point: no gcloud when a key is set


def test_token_falls_back_to_gcloud_without_credentials():
    d = _Dummy()
    d._credentials = None
    d._access_token = None
    d._token_expiry = 0
    result = MagicMock()
    result.stdout = "gcloud-token\n"
    with patch(
        "esperanto.providers.vertex_auth.subprocess.run", return_value=result
    ) as mock_run:
        token = d._get_access_token()
    assert token == "gcloud-token"
    mock_run.assert_called_once()


def test_gcloud_missing_raises_clear_error():
    """gcloud not installed (container) → clear error naming the key options."""
    d = _Dummy()
    d._credentials = None
    d._access_token = None
    d._token_expiry = 0
    with patch(
        "esperanto.providers.vertex_auth.subprocess.run",
        side_effect=FileNotFoundError("gcloud"),
    ):
        with pytest.raises(RuntimeError, match="credentials_file / credentials_path"):
            d._get_access_token()


# --- end-to-end at construction (the actual bug) --------------------------- #


def test_tts_construction_consumes_credentials_path(monkeypatch, fake_creds):
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    from esperanto.providers.tts.vertex import VertexTextToSpeechModel

    with patch(
        "google.oauth2.service_account.Credentials.from_service_account_file",
        return_value=fake_creds,
    ) as mock_load:
        model = VertexTextToSpeechModel(
            model_name="standard", config={"credentials_path": "/fake/key.json"}
        )
    mock_load.assert_called_once()
    assert model._credentials is fake_creds


def test_embedding_construction_consumes_credentials_path(monkeypatch, fake_creds):
    monkeypatch.setenv("VERTEX_PROJECT", "test-project")
    from esperanto.providers.embedding.vertex import VertexEmbeddingModel

    with patch(
        "google.oauth2.service_account.Credentials.from_service_account_file",
        return_value=fake_creds,
    ) as mock_load:
        model = VertexEmbeddingModel(config={"credentials_path": "/fake/key.json"})
    mock_load.assert_called_once()
    assert model._credentials is fake_creds
