use std::fmt;

use regex::Regex;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

/// A validated kernel name matching `^[a-z][-a-z0-9]*[a-z0-9]$`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelName(String);

impl KernelName {
    pub fn new(name: impl Into<String>) -> Result<Self, KernelNameError> {
        let name = name.into();
        let pattern = Regex::new(r"^[a-z][-a-z0-9]*[a-z0-9]$").unwrap();

        if !pattern.is_match(&name) {
            return Err(KernelNameError(name));
        }

        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn python_name(&self) -> String {
        self.0.replace("-", "_")
    }
}

impl AsRef<str> for KernelName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KernelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct KernelNameError(String);

impl fmt::Display for KernelNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Invalid kernel name `{}`. Name must:\n\
             - Start with a lowercase letter (a-z)\n\
             - Contain only lowercase letters, digits, and dashes\n\
             - End with a lowercase letter or digit\n\
             - Be at least 2 characters long\n\
             Examples: `my-kernel`, `relu2d`, `flash-attention`",
            self.0
        )
    }
}

impl std::error::Error for KernelNameError {}

impl<'de> Deserialize<'de> for KernelName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        KernelName::new(s).map_err(de::Error::custom)
    }
}

impl Serialize for KernelName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_name_valid() {
        assert!(KernelName::new("my-kernel").is_ok());
        assert!(KernelName::new("relu2d").is_ok());
        assert!(KernelName::new("flash-attention").is_ok());
        assert!(KernelName::new("a1").is_ok());
        assert!(KernelName::new("ab").is_ok());
        assert!(KernelName::new("my--kernel").is_ok());
    }

    #[test]
    fn test_kernel_name_invalid() {
        assert!(KernelName::new("my_kernel").is_err());
        assert!(KernelName::new("MyKernel").is_err());
        assert!(KernelName::new("a").is_err());
        assert!(KernelName::new("my-kernel-").is_err());
        assert!(KernelName::new("-my-kernel").is_err());
        assert!(KernelName::new("1kernel").is_err());
        assert!(KernelName::new("").is_err());
    }
}
