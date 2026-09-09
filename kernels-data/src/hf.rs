//! Construction of Hugging Face Hub clients.
//!
//! The `hf-hub` crate resolves nothing from the environment: its builder
//! falls back to the public endpoint, no token, and a *relative*
//! `.cache/huggingface/hub` directory. Earlier versions mirrored the Python
//! `huggingface_hub` package, but that resolution moved into the `hfrs`
//! command-line front-end, so it is reimplemented here — with the addition of
//! the `KERNELS_CACHE` override.
//!
//! Proxies need no handling: `hf-hub` builds a default `reqwest` client, which
//! already honours `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, and `NO_PROXY`.
//!
//! [`offline`], [`etag_timeout`], and [`download_timeout`] are resolved here
//! but cannot be handed to `hf-hub`, which has no knob for any of them.
//! Callers have to consult them, exactly as `huggingface_hub.is_offline_mode`
//! expects its callers to.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use hf_hub::{HFClient, HFError};
use thiserror::Error;

use crate::cache::{self, env_path, env_string};

/// Default Hub endpoint, matching `huggingface_hub`.
const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// Name of the token file inside the Hugging Face home directory.
const TOKEN_FILENAME: &str = "token";

/// Default timeout for both etag and download requests, as in
/// `huggingface_hub`.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Error building a Hub client.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HFKernelsClientError {
    /// The kernel cache directory could not be determined, so the client
    /// would fall back to a directory relative to the working directory.
    #[error(
        "cannot determine the kernel cache directory, set `KERNELS_CACHE`, `HF_HUB_CACHE`, or `HF_HOME`"
    )]
    UnknownCacheDir,

    /// The underlying `hf-hub` client could not be constructed.
    #[error("cannot create Hugging Face Hub client")]
    Client(#[from] HFError),
}

/// Builder for a Hub client configured the way `huggingface_hub` configures
/// itself, so that Rust and Python talk to the same endpoint with the same
/// credentials and share one kernel cache.
///
/// Values set on the builder always win. Anything left unset is resolved from
/// the environment at build time:
///
/// | Setting | Resolution |
/// |---|---|
/// | Endpoint | `HF_ENDPOINT`, else `https://huggingface.co` |
/// | Token | `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, else the token file at `HF_TOKEN_PATH` or `$HF_HOME/token`; skipped entirely when `HF_HUB_DISABLE_IMPLICIT_TOKEN` is truthy |
/// | Cache directory | [`cache::kernels_cache`] |
/// | User agent | `kernels/<version>`, with `HF_HUB_USER_AGENT_ORIGIN` appended |
///
/// Offline mode and the request timeouts are not client settings, since
/// `hf-hub` cannot represent them; see [`offline`], [`etag_timeout`], and
/// [`download_timeout`].
///
/// Unlike `hf-hub`'s own builder, the cache directory is never left to a
/// working-directory-relative default: building fails instead.
#[derive(Clone, Debug, Default)]
pub struct HFKernelsClientBuilder {
    endpoint: Option<String>,
    token: Option<String>,
    cache_dir: Option<PathBuf>,
    user_agent: Option<String>,
}

impl HFKernelsClientBuilder {
    /// A builder with everything resolved from the environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the Hub endpoint.
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Override the authentication token, bypassing all implicit lookups.
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Override the cache directory that kernels are downloaded into.
    pub fn cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }

    /// Override the `User-Agent` header.
    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// Build an asynchronous client.
    pub fn build(self) -> Result<HFClient, HFKernelsClientError> {
        Ok(self.hf_hub_builder()?.build()?)
    }

    /// Build a blocking client.
    #[cfg(feature = "blocking")]
    pub fn build_sync(self) -> Result<hf_hub::HFClientSync, HFKernelsClientError> {
        Ok(self.hf_hub_builder()?.build_sync()?)
    }

    fn hf_hub_builder(self) -> Result<hf_hub::HFClientBuilder, HFKernelsClientError> {
        let cache_dir = match self.cache_dir {
            Some(cache_dir) => cache_dir,
            None => cache::kernels_cache().ok_or(HFKernelsClientError::UnknownCacheDir)?,
        };

        let mut builder = HFClient::builder()
            .endpoint(select_endpoint(self.endpoint, env_string("HF_ENDPOINT")))
            .cache_dir(cache_dir)
            .user_agent(self.user_agent.unwrap_or_else(default_user_agent));

        // `hf-hub` has no way to unset a token, so only set it when we have
        // one.
        if let Some(token) = resolve_token(self.token) {
            builder = builder.token(token);
        }

        Ok(builder)
    }
}

/// Whether the Hub must be treated as unreachable.
///
/// Resolves `HF_HUB_OFFLINE`, falling back to `TRANSFORMERS_OFFLINE` like
/// `huggingface_hub` does. Nothing enforces this: `hf-hub` has no offline
/// mode, so callers must check it and serve from the local cache instead of
/// making requests.
pub fn offline() -> bool {
    env_bool("HF_HUB_OFFLINE") || env_bool("TRANSFORMERS_OFFLINE")
}

/// Timeout for the metadata requests that resolve a revision to an etag
/// (`HF_HUB_ETAG_TIMEOUT`).
pub fn etag_timeout() -> Duration {
    env_timeout("HF_HUB_ETAG_TIMEOUT")
}

/// Timeout for download requests (`HF_HUB_DOWNLOAD_TIMEOUT`).
pub fn download_timeout() -> Duration {
    env_timeout("HF_HUB_DOWNLOAD_TIMEOUT")
}

/// The `User-Agent` announcing this library, matching the `library_name` and
/// `library_version` that the Python package reports, plus the
/// `HF_HUB_USER_AGENT_ORIGIN` attribution when set.
fn default_user_agent() -> String {
    let version = env!("CARGO_PKG_VERSION");
    match env_string("HF_HUB_USER_AGENT_ORIGIN") {
        Some(origin) => format!("kernels/{version}; {origin}"),
        None => format!("kernels/{version}"),
    }
}

/// A boolean environment variable, using the same truthy values as
/// `huggingface_hub`: anything else (including `0` and `false`) is false.
fn env_bool(var: &str) -> bool {
    is_true(env_string(var).as_deref())
}

fn is_true(value: Option<&str>) -> bool {
    matches!(
        value.map(str::to_ascii_uppercase).as_deref(),
        Some("1" | "ON" | "YES" | "TRUE")
    )
}

/// A timeout in whole seconds. Zero and unparseable values fall back to the
/// default, where `huggingface_hub` would raise on the latter.
fn env_timeout(var: &str) -> Duration {
    select_timeout(env_string(var).as_deref())
}

fn select_timeout(value: Option<&str>) -> Duration {
    value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map_or(DEFAULT_TIMEOUT, Duration::from_secs)
}

/// Resolve the token from the environment and the token file.
fn resolve_token(explicit: Option<String>) -> Option<String> {
    select_token(
        explicit,
        env_bool("HF_HUB_DISABLE_IMPLICIT_TOKEN"),
        env_string("HF_TOKEN"),
        env_string("HUGGING_FACE_HUB_TOKEN"),
        token_path().as_deref(),
    )
}

/// Path of the token file: `HF_TOKEN_PATH`, else `token` in the Hugging Face
/// home directory.
fn token_path() -> Option<PathBuf> {
    env_path("HF_TOKEN_PATH").or_else(|| Some(cache::hf_home()?.join(TOKEN_FILENAME)))
}

/// A token stored in a file; blank files count as absent.
fn read_token_file(path: &Path) -> Option<String> {
    let token = fs::read_to_string(path).ok()?.trim().to_string();
    (!token.is_empty()).then_some(token)
}

// As in `cache`, the resolution rules are separated from the environment
// lookups so that they can be tested without mutating the process
// environment.

fn select_endpoint(explicit: Option<String>, hf_endpoint: Option<String>) -> String {
    explicit
        .or(hf_endpoint)
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string())
}

fn select_token(
    explicit: Option<String>,
    disable_implicit_token: bool,
    hf_token: Option<String>,
    huggingface_hub_token: Option<String>,
    token_file: Option<&Path>,
) -> Option<String> {
    // An explicitly configured token is not an implicit one, so it is used
    // even when implicit tokens are disabled.
    if explicit.is_some() {
        return explicit;
    }
    if disable_implicit_token {
        return None;
    }
    hf_token
        .or(huggingface_hub_token)
        .or_else(|| read_token_file(token_file?))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    fn text(value: &str) -> Option<String> {
        Some(value.to_string())
    }

    #[test]
    fn endpoint_precedence() {
        assert_eq!(
            select_endpoint(text("https://explicit"), text("https://env")),
            "https://explicit"
        );
        assert_eq!(select_endpoint(None, text("https://env")), "https://env");
        assert_eq!(select_endpoint(None, None), DEFAULT_ENDPOINT);
    }

    #[test]
    fn token_precedence() {
        assert_eq!(
            select_token(text("explicit"), false, text("hf"), text("legacy"), None),
            text("explicit")
        );
        assert_eq!(
            select_token(None, false, text("hf"), text("legacy"), None),
            text("hf")
        );
        assert_eq!(
            select_token(None, false, None, text("legacy"), None),
            text("legacy")
        );
        assert_eq!(select_token(None, false, None, None, None), None);
    }

    /// Disabling implicit tokens must not discard a token the caller passed
    /// in deliberately.
    #[test]
    fn disabling_implicit_tokens_keeps_explicit_token() {
        assert_eq!(
            select_token(text("explicit"), true, text("hf"), None, None),
            text("explicit")
        );
        assert_eq!(select_token(None, true, text("hf"), None, None), None);
    }

    #[test]
    fn token_falls_back_to_token_file() -> std::io::Result<()> {
        let dir = TempDir::new()?;
        let token_file = dir.path().join("token");
        fs::write(&token_file, "  file-token\n")?;

        assert_eq!(
            select_token(None, false, None, None, Some(&token_file)),
            text("file-token")
        );
        // The environment still wins over the file.
        assert_eq!(
            select_token(None, false, text("hf"), None, Some(&token_file)),
            text("hf")
        );
        Ok(())
    }

    #[test]
    fn blank_and_missing_token_files_are_ignored() -> std::io::Result<()> {
        let dir = TempDir::new()?;
        let blank = dir.path().join("blank");
        fs::write(&blank, "  \n")?;

        assert_eq!(select_token(None, false, None, None, Some(&blank)), None);
        assert_eq!(
            select_token(None, false, None, None, Some(&dir.path().join("missing"))),
            None
        );
        Ok(())
    }

    /// Only `huggingface_hub`'s truthy values count, so `HF_HUB_OFFLINE=0`
    /// does not switch offline mode on.
    #[test]
    fn only_documented_values_are_true() {
        for value in ["1", "ON", "on", "YES", "yes", "TRUE", "true", "True"] {
            assert!(is_true(Some(value)), "{value} should be true");
        }
        for value in ["0", "OFF", "NO", "FALSE", "false", "AUTO", "", "maybe"] {
            assert!(!is_true(Some(value)), "{value} should be false");
        }
        assert!(!is_true(None));
    }

    #[test]
    fn timeouts_fall_back_to_the_default() {
        assert_eq!(select_timeout(Some("30")), Duration::from_secs(30));
        assert_eq!(select_timeout(Some(" 30 ")), Duration::from_secs(30));
        // Zero is falsy in `huggingface_hub`, so it means "unset".
        assert_eq!(select_timeout(Some("0")), DEFAULT_TIMEOUT);
        assert_eq!(select_timeout(Some("-1")), DEFAULT_TIMEOUT);
        assert_eq!(select_timeout(Some("soon")), DEFAULT_TIMEOUT);
        assert_eq!(select_timeout(None), DEFAULT_TIMEOUT);
    }

    /// The cache directory must never silently become relative to the
    /// working directory, the way `hf-hub`'s own default does.
    #[test]
    fn explicit_cache_dir_is_used() {
        let builder = HFKernelsClientBuilder::new()
            .cache_dir("/tmp/kernels-cache")
            .token("test-token");
        let client = builder.build().expect("client should build");
        assert_eq!(client.cache_dir(), Path::new("/tmp/kernels-cache"));
    }
}
