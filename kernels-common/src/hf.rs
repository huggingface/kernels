//! Construction of Hugging Face Hub clients.
//!
//! The hf-hub crate stopped handling standard huggingface_hub environment
//! variables. This module adds a wrapper for the builder that adds back
//! environment variable support, as closely aligned to huggingface_hub
//! as possible.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use hf_hub::{HFClient, HFError};
use thiserror::Error;

/// Default Hub endpoint, matching `huggingface_hub`.
const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// Name of the token file inside the Hugging Face home directory.
const TOKEN_FILENAME: &str = "token";

/// Environment variable interpreted as a bool.
fn env_bool(var: &str) -> bool {
    is_true(env_string(var).as_deref())
}

/// Truthy values.
fn is_true(value: Option<&str>) -> bool {
    // See: https://github.com/huggingface/huggingface_hub/blob/f14866648507aa58afc9554c712a4e1f70dd5c3e/src/huggingface_hub/constants.py#L12
    matches!(
        value.map(str::to_ascii_uppercase).as_deref(),
        Some("1" | "ON" | "YES" | "TRUE")
    )
}

/// Environment variable interpreted as a string.
///
/// Empty values count as unset. A leading `~` is expmanded to the user's
/// home directory.
fn env_path(var: &str) -> Option<PathBuf> {
    env::var_os(var)
        .filter(|value| !value.is_empty())
        .map(|value| expand_tilde(PathBuf::from(value)))
}

/// Environment variable interpreted as a string.
///
/// An empty value counts as unset.
fn env_string(var: &str) -> Option<String> {
    env::var(var).ok().filter(|value| !value.is_empty())
}

/// Expand a leading `~` to the home directory.
pub(crate) fn expand_tilde(path: PathBuf) -> PathBuf {
    expand_tilde_with_home(path, env::home_dir())
}

fn expand_tilde_with_home(path: PathBuf, home: Option<PathBuf>) -> PathBuf {
    // Normally one would use `expand_tilde`. But this is a variant
    // that takes a home directory to make it testable.

    // Only matches a whole leading `~` component, so `~user/x` does not.
    let Ok(rest) = path.strip_prefix("~") else {
        return path;
    };
    let Some(home) = home else {
        return path;
    };
    if rest == Path::new("") {
        return home;
    }
    home.join(rest)
}

/// Error building a Hub client.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HFKernelsClientError {
    /// The kernel cache directory could not be detepmined.
    #[error(
        "cannot determine the kernel cache directory, set `KERNELS_CACHE`, `HF_HUB_CACHE`, or `HF_HOME`"
    )]
    UnknownCacheDir,

    /// The underlying `hf-hub` client could not be constructed.
    #[error("cannot create Hugging Face Hub client")]
    Client(#[from] HFError),
}

/// Hugging Face Hub client builder, with standard huggingface-hub environment variable support.
#[derive(Clone, Debug, Default)]
pub struct HFKernelsClientBuilder {
    endpoint: Option<String>,
    token: Option<String>,
    cache_dir: Option<PathBuf>,
    user_agent: Option<String>,
}

impl HFKernelsClientBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the Hub endpoint.
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Override the authentication token.
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Override the cache directory.
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
        Ok(self.hf_client_builder()?.build()?)
    }

    /// Build a blocking client.
    pub fn build_sync(self) -> Result<hf_hub::HFClientSync, HFKernelsClientError> {
        Ok(self.hf_client_builder()?.build_sync()?)
    }

    fn hf_client_builder(self) -> Result<hf_hub::HFClientBuilder, HFKernelsClientError> {
        let cache_dir = match self.cache_dir {
            Some(cache_dir) => cache_dir,
            None => kernels_cache().ok_or(HFKernelsClientError::UnknownCacheDir)?,
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

/// The Hugging Face home directory.
fn hf_home() -> Option<PathBuf> {
    resolve_hf_home(
        env_path("HF_HOME"),
        env_path("XDG_CACHE_HOME"),
        env::home_dir(),
    )
}

/// The Hub cache directory.
fn hf_hub_cache() -> Option<PathBuf> {
    resolve_hf_hub_cache(
        env_path("HF_HUB_CACHE"),
        env_path("HUGGINGFACE_HUB_CACHE"),
        hf_home(),
    )
}

/// The kernels cache directory.
pub(crate) fn kernels_cache() -> Option<PathBuf> {
    resolve_kernels_cache(env_path("KERNELS_CACHE"), hf_hub_cache())
}

fn resolve_hf_home(
    hf_home: Option<PathBuf>,
    xdg_cache_home: Option<PathBuf>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(hf_home) = hf_home {
        return Some(hf_home);
    }
    let cache_home = match xdg_cache_home {
        Some(xdg_cache_home) => xdg_cache_home,
        None => home?.join(".cache"),
    };
    Some(cache_home.join("huggingface"))
}

fn resolve_hf_hub_cache(
    hf_hub_cache: Option<PathBuf>,
    huggingface_hub_cache: Option<PathBuf>,
    hf_home: Option<PathBuf>,
) -> Option<PathBuf> {
    hf_hub_cache
        .or(huggingface_hub_cache)
        .or_else(|| Some(hf_home?.join("hub")))
}

fn resolve_kernels_cache(
    kernels_cache: Option<PathBuf>,
    hf_hub_cache: Option<PathBuf>,
) -> Option<PathBuf> {
    kernels_cache.or(hf_hub_cache)
}

/// The default user agent.
fn default_user_agent() -> String {
    let version = env!("CARGO_PKG_VERSION");
    match env_string("HF_HUB_USER_AGENT_ORIGIN") {
        Some(origin) => format!("kernels/{version}; {origin}"),
        None => format!("kernels/{version}"),
    }
}

/// Select the endpoint to use.
fn select_endpoint(explicit: Option<String>, hf_endpoint: Option<String>) -> String {
    explicit
        .or(hf_endpoint)
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string())
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

/// Path of the token file.
fn token_path() -> Option<PathBuf> {
    env_path("HF_TOKEN_PATH").or_else(|| Some(hf_home()?.join(TOKEN_FILENAME)))
}

/// A token stored in a file; blank files count as absent.
fn read_token_file(path: &Path) -> Option<String> {
    let token = fs::read_to_string(path).ok()?.trim().to_string();
    (!token.is_empty()).then_some(token)
}

/// Select the token to use.
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

    fn path(path: &str) -> Option<PathBuf> {
        Some(PathBuf::from(path))
    }

    #[test]
    fn hf_home_prefers_explicit_setting() {
        assert_eq!(
            resolve_hf_home(path("/hf-home"), path("/xdg"), path("/home/user")),
            path("/hf-home")
        );
    }

    #[test]
    fn hf_home_falls_back_to_xdg_cache_home() {
        assert_eq!(
            resolve_hf_home(None, path("/xdg"), path("/home/user")),
            path("/xdg/huggingface")
        );
    }

    #[test]
    fn hf_home_falls_back_to_home_directory() {
        assert_eq!(
            resolve_hf_home(None, None, path("/home/user")),
            path("/home/user/.cache/huggingface")
        );
    }

    #[test]
    fn hf_home_is_unknown_without_home_directory() {
        assert_eq!(resolve_hf_home(None, None, None), None);
    }

    #[test]
    fn hub_cache_precedence() {
        assert_eq!(
            resolve_hf_hub_cache(path("/hub"), path("/legacy"), path("/hf-home")),
            path("/hub")
        );
        assert_eq!(
            resolve_hf_hub_cache(None, path("/legacy"), path("/hf-home")),
            path("/legacy")
        );
        assert_eq!(
            resolve_hf_hub_cache(None, None, path("/hf-home")),
            path("/hf-home/hub")
        );
        assert_eq!(resolve_hf_hub_cache(None, None, None), None);
    }

    #[test]
    fn kernels_cache_overrides_hub_cache() {
        assert_eq!(
            resolve_kernels_cache(path("/kernels"), path("/hub")),
            path("/kernels")
        );
        assert_eq!(resolve_kernels_cache(None, path("/hub")), path("/hub"));
        assert_eq!(resolve_kernels_cache(None, None), None);
    }

    /// Expands `input` against a fixed home directory.
    fn expanded(input: &str) -> PathBuf {
        expand_tilde_with_home(PathBuf::from(input), path("/home/user"))
    }

    #[test]
    fn tilde_is_expanded_to_the_home_directory() {
        assert_eq!(expanded("~"), PathBuf::from("/home/user"));
        assert_eq!(
            expanded("~/.cache/hub"),
            PathBuf::from("/home/user/.cache/hub")
        );
    }

    #[test]
    fn only_a_leading_tilde_component_is_expanded() {
        // Another user's home directory needs a passwd lookup.
        assert_eq!(expanded("~other/hub"), PathBuf::from("~other/hub"));
        // A tilde that is not leading is part of the name.
        assert_eq!(expanded("/cache/~/hub"), PathBuf::from("/cache/~/hub"));
        assert_eq!(expanded("/absolute/hub"), PathBuf::from("/absolute/hub"));
        assert_eq!(expanded("relative/hub"), PathBuf::from("relative/hub"));
    }

    /// `expanduser` leaves the path alone when there is no home directory,
    /// rather than failing.
    #[test]
    fn tilde_is_kept_without_a_home_directory() {
        assert_eq!(
            expand_tilde_with_home(PathBuf::from("~/hub"), None),
            PathBuf::from("~/hub")
        );
    }

    /// An explicitly empty variable must not resolve to a relative path.
    #[test]
    fn empty_variables_count_as_unset() {
        // SAFETY: single-threaded access to this variable; no other test
        // reads or writes it.
        unsafe { env::set_var("KERNELS_DATA_TEST_EMPTY", "") };
        assert_eq!(env_path("KERNELS_DATA_TEST_EMPTY"), None);

        unsafe { env::set_var("KERNELS_DATA_TEST_EMPTY", "/value") };
        assert_eq!(env_path("KERNELS_DATA_TEST_EMPTY"), path("/value"));

        unsafe { env::remove_var("KERNELS_DATA_TEST_EMPTY") };
        assert_eq!(env_path("KERNELS_DATA_TEST_EMPTY"), None);
    }
}
