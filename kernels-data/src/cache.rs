//! Resolution of the cache directories that kernels are loaded from.
//!
//! Mirrors `huggingface_hub.constants` and `kernels.hf_hub`, so that Rust and
//! Python agree on where kernels live. The `hf-hub` crate does not resolve
//! these variables (only its `hfrs` command-line front-end does), so the
//! chain is reimplemented here.
//!
//! Every variable is treated as unset when it is empty, matching the falsy
//! handling on the Python side, and a leading `~` is expanded as Python's
//! `os.path.expanduser` would.

use std::env;
use std::path::{Path, PathBuf};

/// The Hugging Face home directory.
///
/// `HF_HOME` when set, otherwise `huggingface` inside the XDG cache
/// directory (`XDG_CACHE_HOME`, defaulting to `~/.cache`).
///
/// Returns `None` when the fallback applies but the home directory cannot be
/// determined.
pub fn hf_home() -> Option<PathBuf> {
    resolve_hf_home(
        env_path("HF_HOME"),
        env_path("XDG_CACHE_HOME"),
        env::home_dir(),
    )
}

/// The Hub cache directory, which kernels are downloaded into.
///
/// `HF_HUB_CACHE` when set, then the legacy `HUGGINGFACE_HUB_CACHE`, then
/// `hub` inside [`hf_home`].
pub fn hf_hub_cache() -> Option<PathBuf> {
    resolve_hf_hub_cache(
        env_path("HF_HUB_CACHE"),
        env_path("HUGGINGFACE_HUB_CACHE"),
        hf_home(),
    )
}

/// The cache directory that kernels are loaded from.
///
/// `KERNELS_CACHE` when set, otherwise the [`hf_hub_cache`].
pub fn kernels_cache() -> Option<PathBuf> {
    resolve_kernels_cache(env_path("KERNELS_CACHE"), hf_hub_cache())
}

/// A path-valued environment variable; empty values count as unset and a
/// leading `~` is expanded.
pub(crate) fn env_path(var: &str) -> Option<PathBuf> {
    env::var_os(var)
        .filter(|value| !value.is_empty())
        .map(|value| expand_tilde(PathBuf::from(value)))
}

/// A string-valued environment variable; empty values count as unset.
pub(crate) fn env_string(var: &str) -> Option<String> {
    env::var(var).ok().filter(|value| !value.is_empty())
}

/// Expand a leading `~` to the home directory, like Python's
/// `os.path.expanduser`.
///
/// The Hub variables are documented as accepting `~`, and expanding it here
/// avoids creating a directory literally named `~`.
pub(crate) fn expand_tilde(path: PathBuf) -> PathBuf {
    expand_tilde_in(path, env::home_dir())
}

/// `~user` forms are deliberately left alone: resolving another user's home
/// directory needs a passwd lookup, and these variables are not set that way
/// in practice. A path is also returned unchanged when the home directory is
/// unknown, which is what `expanduser` does too.
fn expand_tilde_in(path: PathBuf, home: Option<PathBuf>) -> PathBuf {
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

// The resolution rules are split out from the environment lookups so that
// they can be tested without mutating the process environment, which is
// global (and `unsafe`) and would race with tests running in parallel.

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

#[cfg(test)]
mod tests {
    use super::*;

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
        expand_tilde_in(PathBuf::from(input), path("/home/user"))
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
            expand_tilde_in(PathBuf::from("~/hub"), None),
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
