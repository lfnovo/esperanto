# Feature Request: SSL Verification Configuration

## Problem Statement

Users running local LLM providers (Ollama, LMStudio, etc.) behind reverse proxies with self-signed SSL certificates cannot connect to these services. The HTTP clients used by Esperanto verify SSL certificates by default, and there's no way to:

1. Disable SSL verification for testing/development environments
2. Specify a custom CA certificate bundle for self-signed certificates

### User Report

From [open-notebook#274](https://github.com/lfnovo/open-notebook/issues/274):

- User has Ollama and LMStudio running behind a Caddy reverse proxy with self-signed certificates
- Error for Ollama: `[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate`
- Error for LMStudio (OpenAI-compatible): `Connection error`
- Workaround: Using HTTP instead of HTTPS works, but is not ideal for security

### Technical Root Cause

In `esperanto/src/esperanto/providers/llm/base.py` (and other base classes), HTTP clients are created without SSL configuration:

```python
def _create_http_clients(self) -> None:
    import httpx
    timeout = self._get_timeout()
    self.client = httpx.Client(timeout=timeout)
    self.async_client = httpx.AsyncClient(timeout=timeout)
```

Python's SSL verification uses the `certifi` package certificate store, not the system's certificate store. Even after adding custom CA certificates to the container's system store, Python doesn't pick them up.

## Affected Components

The following base classes create HTTP clients that need SSL configuration:

| File | Lines |
|------|-------|
| `providers/llm/base.py` | 190-191 |
| `providers/embedding/base.py` | 351-352 |
| `providers/stt/base.py` | 122-123 |
| `providers/tts/base.py` | 163-164 |
| `providers/reranker/base.py` | 106-107 |

## Proposed Solution

### Configuration Options

Add two new configuration parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verify_ssl` | `bool` | `True` | Enable/disable SSL certificate verification |
| `ssl_ca_bundle` | `str \| None` | `None` | Path to custom CA certificate bundle file |

### Environment Variables

Following the pattern established by `TimeoutMixin`:

| Variable | Description |
|----------|-------------|
| `ESPERANTO_SSL_VERIFY` | Set to `"false"` to disable SSL verification globally |
| `ESPERANTO_SSL_CA_BUNDLE` | Path to custom CA bundle file |

### Priority Hierarchy

Similar to timeout configuration:

1. **Config dict** (highest priority): `config={"verify_ssl": False}`
2. **Environment variable**: `ESPERANTO_SSL_VERIFY=false`
3. **Default value** (lowest priority): `True`

## Implementation Options

### Option A: Create SSLMixin (Recommended)

Create a new `SSLMixin` class similar to `TimeoutMixin`:

```python
# esperanto/src/esperanto/utils/ssl.py

import os
from typing import Union

class SSLMixin:
    """Mixin providing SSL configuration functionality."""

    def _get_ssl_verify(self) -> Union[bool, str]:
        """Get SSL verification setting.

        Returns:
            bool: True/False for enable/disable verification
            str: Path to CA bundle file if ssl_ca_bundle is set
        """
        # 1. Config dict (highest priority)
        if hasattr(self, '_config'):
            if "ssl_ca_bundle" in self._config and self._config["ssl_ca_bundle"]:
                return self._config["ssl_ca_bundle"]
            if "verify_ssl" in self._config:
                return self._config["verify_ssl"]

        # 2. Environment variables
        ca_bundle = os.getenv("ESPERANTO_SSL_CA_BUNDLE")
        if ca_bundle:
            return ca_bundle

        verify_env = os.getenv("ESPERANTO_SSL_VERIFY", "").lower()
        if verify_env in ("false", "0", "no"):
            return False

        # 3. Default: SSL verification enabled
        return True
```

Update `_create_http_clients` in base classes:

```python
def _create_http_clients(self) -> None:
    import httpx
    timeout = self._get_timeout()
    verify = self._get_ssl_verify()
    self.client = httpx.Client(timeout=timeout, verify=verify)
    self.async_client = httpx.AsyncClient(timeout=timeout, verify=verify)
```

**Pros:**
- Clean separation of concerns
- Consistent with existing `TimeoutMixin` pattern
- Easy to test in isolation
- Single source of truth for SSL configuration logic

**Cons:**
- Requires updating all 5 base classes to inherit from the new mixin
- Slightly more complex implementation

### Option B: Direct Implementation

Add SSL configuration directly to each base class without a mixin:

```python
# In each base class

verify_ssl: bool = True
ssl_ca_bundle: Optional[str] = None

def _create_http_clients(self) -> None:
    import httpx
    import os

    timeout = self._get_timeout()

    # Determine SSL verification setting
    verify: Union[bool, str] = True

    # Check config
    if self.ssl_ca_bundle:
        verify = self.ssl_ca_bundle
    elif not self.verify_ssl:
        verify = False
    # Check environment variables
    elif os.getenv("ESPERANTO_SSL_CA_BUNDLE"):
        verify = os.getenv("ESPERANTO_SSL_CA_BUNDLE")
    elif os.getenv("ESPERANTO_SSL_VERIFY", "").lower() in ("false", "0", "no"):
        verify = False

    self.client = httpx.Client(timeout=timeout, verify=verify)
    self.async_client = httpx.AsyncClient(timeout=timeout, verify=verify)
```

**Pros:**
- Simpler, faster to implement
- No new files needed

**Cons:**
- Code duplication across 5 base classes
- Harder to maintain consistency
- Doesn't follow established patterns

## Usage Examples

### Disable SSL Verification (Development/Testing)

```python
# Via config
model = AIFactory.create_language(
    provider="ollama",
    model_name="llama3",
    config={"verify_ssl": False}
)

# Via environment variable
# export ESPERANTO_SSL_VERIFY=false
```

### Custom CA Bundle (Self-Signed Certificates)

```python
# Via config
model = AIFactory.create_language(
    provider="ollama",
    model_name="llama3",
    config={"ssl_ca_bundle": "/path/to/ca-bundle.pem"}
)

# Via environment variable
# export ESPERANTO_SSL_CA_BUNDLE=/path/to/ca-bundle.pem
```

## Security Considerations

- SSL verification should remain **enabled by default**
- Documentation should warn users about the security implications of disabling SSL verification
- The `ssl_ca_bundle` option is the preferred solution for self-signed certificates (maintains security while allowing custom CAs)

## Testing Requirements

1. Unit tests for SSL configuration priority hierarchy
2. Integration tests with mocked HTTPS endpoints
3. Test cases:
   - Default behavior (SSL enabled)
   - `verify_ssl=False` disables verification
   - `ssl_ca_bundle` uses custom CA
   - Environment variables override defaults
   - Config takes precedence over environment variables

## Documentation Updates

- Add SSL configuration section to provider documentation
- Update README with examples
- Add security warning about disabling SSL verification
