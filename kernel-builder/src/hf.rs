use eyre::{Context, Result};

/// Build a tokio runtime for one-shot async calls.
pub fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Cannot create async runtime")
}

/// Build an HF API client.
pub fn api() -> Result<huggingface_hub::HfApi> {
    huggingface_hub::HfApi::new().context("Cannot create Hugging Face API client")
}

/// Resolve the HF username of the currently logged-in user via `whoami`.
/// Requires a valid HF token to be configured.
pub fn whoami_username() -> Result<String> {
    let rt = runtime()?;
    let api = api()?;

    rt.block_on(async { api.whoami().await.map(|user| user.username) })
        .map_err(|_| {
            eyre::eyre!(
                "Not logged in to Hugging Face. Run `hf auth login` first, \
                 or use --name <owner/repo> to skip auto-detection."
            )
        })
}
