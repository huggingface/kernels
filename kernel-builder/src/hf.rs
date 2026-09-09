use eyre::{Context, Result};
use hf_hub::{HFClientSync, HFRepositorySync, RepoType};
use kernels_data::hf::HFKernelsClientBuilder;

/// Build a sync HF API client.
///
/// Endpoint, token, and cache directory are resolved from the environment the
/// way `huggingface_hub` resolves them; see [`HFKernelsClientBuilder`].
pub fn api() -> Result<HFClientSync> {
    HFKernelsClientBuilder::new()
        .build_sync()
        .context("Cannot create Hugging Face API client")
}

/// Get a repo handle.
pub fn repo_handle<T: RepoType>(api: &HFClientSync, repo_id: &str) -> HFRepositorySync<T> {
    let parts: Vec<&str> = repo_id.splitn(2, '/').collect();
    if parts.len() == 2 {
        api.repository(T::default(), parts[0], parts[1])
    } else {
        api.repository(T::default(), "", repo_id)
    }
}

/// Resolve the HF username of the currently logged-in user via `whoami`.
/// Requires a valid HF token to be configured.
pub fn whoami_username() -> Result<String> {
    api()?
        .whoami()
        .send()
        .map(|user| user.username)
        .map_err(|_| {
            eyre::eyre!(
                "Not logged in to Hugging Face. Run `hf auth login` first, \
                     or use --name <owner/repo> to skip auto-detection."
            )
        })
}
