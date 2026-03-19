use std::{fs, path::PathBuf};

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

/// Resolve the HF access token using the standard resolution order:
/// `HF_TOKEN` env → `HF_TOKEN_PATH` file → `$HF_HOME/token` → `~/.cache/huggingface/token`.
pub fn token() -> Option<String> {
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            return Some(token);
        }
    }

    let token_path = std::env::var("HF_TOKEN_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let hf_home = std::env::var("HF_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    let home = std::env::var("HOME")
                        .or_else(|_| std::env::var("USERPROFILE"))
                        .unwrap_or_default();
                    PathBuf::from(home).join(".cache/huggingface")
                });
            hf_home.join("token")
        });

    fs::read_to_string(token_path)
        .ok()
        .map(|t| t.trim().to_owned())
}
