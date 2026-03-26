use eyre::{Context, Result};

/// Build a sync HF API client.
pub fn api() -> Result<huggingface_hub::HfApiSync> {
    huggingface_hub::HfApiSync::new().context("Cannot create Hugging Face API client")
}

/// Resolve the HF username of the currently logged-in user via `whoami`.
/// Requires a valid HF token to be configured.
pub fn whoami_username() -> Result<String> {
    api()?.whoami().map(|user| user.username).map_err(|_| {
        eyre::eyre!(
            "Not logged in to Hugging Face. Run `hf auth login` first, \
                 or use --name <owner/repo> to skip auto-detection."
        )
    })
}
