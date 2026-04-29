use eyre::{Context, Result};
use hf_hub::{HFClientSync, HFRepositorySync, RepoType};

/// Build a sync HF API client.
pub fn api() -> Result<hf_hub::HFClientSync> {
    hf_hub::HFClientSync::new().context("Cannot create Hugging Face API client")
}

/// Get a repo handle.
pub fn repo_handle(api: &HFClientSync, repo_type: RepoType, repo_id: &str) -> HFRepositorySync {
    let parts: Vec<&str> = repo_id.splitn(2, '/').collect();
    if parts.len() == 2 {
        api.repo(repo_type, parts[0], parts[1])
    } else {
        api.repo(repo_type, "", repo_id)
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
