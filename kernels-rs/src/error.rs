#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Kernel(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Library(#[from] libloading::Error),

    #[error(transparent)]
    Hub(#[from] huggingface_hub::HfError),

    #[cfg(feature = "candle")]
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
