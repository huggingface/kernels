//! Utilities for running Nix.

use std::fmt;
use std::path::PathBuf;
use std::process::Command;

use eyre::Result;
use serde::de::{self, Visitor};
use serde::Deserialize;

/// A Nix flake.
pub struct Flake {
    path: PathBuf,
}

impl Flake {
    /// Get a [`Flake`] from a path to a flake directory.
    ///
    /// Returns an error if `flake.nix` is not found in the given path.
    pub fn from_path(path: PathBuf) -> Result<Self> {
        let flake_nix = path.join("flake.nix");
        if !flake_nix.is_file() {
            eyre::bail!("flake.nix not found in {}", path.display());
        }

        Ok(Self { path })
    }
}

/// Nix subcommands.
pub enum NixSubcommand {
    /// Run the program associated with an attribute in a flake.
    #[allow(dead_code)]
    Run {
        flake: Flake,
        attribute: Option<String>,
    },

    /// Spawn a development shell with the given attribute for a flake.
    Develop {
        flake: Flake,
        attribute: Option<String>,
    },
}

/// Nix sandboxing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxMode {
    /// Sandboxing is enabled (strict).
    Enabled,

    /// Sandboxing is relaxed (relaxed).
    Relaxed,

    /// Sandboxing is disabled.
    Disabled,
}

impl<'de> Deserialize<'de> for SandboxMode {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_any(SandboxModeVisitor)
    }
}

struct SandboxModeVisitor;

impl<'de> Visitor<'de> for SandboxModeVisitor {
    type Value = SandboxMode;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a boolean or the string \"relaxed\"")
    }

    fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<SandboxMode, E> {
        Ok(if v {
            SandboxMode::Enabled
        } else {
            SandboxMode::Disabled
        })
    }

    fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<SandboxMode, E> {
        match v {
            "true" => Ok(SandboxMode::Enabled),
            "relaxed" => Ok(SandboxMode::Relaxed),
            "false" => Ok(SandboxMode::Disabled),
            _ => Err(E::unknown_variant(v, &["true", "relaxed", "false"])),
        }
    }
}

/// Setting in the Nix configuration.
#[derive(Deserialize)]
pub struct NixConfigSetting<T> {
    pub value: T,
}

/// Nix configuration.
#[derive(Deserialize)]
pub struct NixConfig {
    pub sandbox: NixConfigSetting<SandboxMode>,
}

impl NixConfig {
    /// Get the sandboxing mode.
    pub fn sandbox_mode(&self) -> SandboxMode {
        self.sandbox.value
    }
}

/// Run Nix commands.
///
/// The Nix struct allows setting Nix options using the builder pattern.
pub struct Nix {
    max_jobs: Option<u32>,
    cores: Option<u32>,
    print_build_logs: bool,
}

impl Nix {
    /// Create a new `Nix` instance.
    pub fn new() -> Self {
        Self {
            max_jobs: None,
            cores: None,
            print_build_logs: false,
        }
    }

    /// Set the maximum number of jobs to run in parallel.
    pub fn max_jobs(mut self, max_jobs: u32) -> Self {
        self.max_jobs = Some(max_jobs);
        self
    }

    /// Set the number of cores to use for parallel jobs.
    pub fn cores(mut self, cores: u32) -> Self {
        self.cores = Some(cores);
        self
    }

    /// Print full build logs on standard error.
    pub fn print_build_logs(mut self, print_build_logs: bool) -> Self {
        self.print_build_logs = print_build_logs;
        self
    }

    /// Get the Nix configuration.
    pub fn config() -> Result<NixConfig> {
        Self::check_installed()?;

        let output = Command::new("nix")
            .args(["show-config", "--json"])
            .output()?;

        if !output.status.success() {
            eyre::bail!("failed to run `nix show-config --json`");
        }

        let config: NixConfig = serde_json::from_slice(&output.stdout)?;

        Ok(config)
    }

    /// Check if Nix is installed and available in the PATH.
    pub fn check_installed() -> Result<()> {
        match Command::new("nix").arg("--version").output() {
            Ok(output) if output.status.success() => Ok(()),
            _ => eyre::bail!(
                "nix is not installed or not found in PATH.\n\
                 Install Nix by running:\n\n    \
                 curl -fsSL https://install.determinate.systems/nix | sh -s -- install\n\n\
                 For more information, visit: https://nixos.org/download"
            ),
        }
    }

    /// Run Nix with the given subcommand.
    pub fn run(self, subcommand: NixSubcommand) -> Result<()> {
        Self::check_installed()?;

        let sandbox = Self::config()?.sandbox_mode();
        if cfg!(target_os = "linux") && sandbox != SandboxMode::Enabled {
            eyre::bail!(
                "Nix sandboxing must be enabled on Linux.\n\
                 Set `sandbox = true` in /etc/nix/nix.conf and restart the Nix daemon."
            );
        }
        if cfg!(target_os = "macos") && sandbox == SandboxMode::Disabled {
            eyre::bail!(
                "Nix sandboxing must be enabled on macOS.\n\
                 Set `sandbox = relaxed` or `sandbox = true` in /etc/nix/nix.conf and restart the Nix daemon."
            );
        }

        let mut cmd = Command::new("nix");

        match &subcommand {
            NixSubcommand::Run { flake, attribute } => {
                cmd.arg("run");
                match attribute {
                    Some(attr) => cmd.arg(format!("{}#{}", flake.path.display(), attr)),
                    None => cmd.arg(flake.path.as_os_str()),
                };
            }
            NixSubcommand::Develop { flake, attribute } => {
                cmd.arg("develop");
                match attribute {
                    Some(attr) => cmd.arg(format!("{}#{}", flake.path.display(), attr)),
                    None => cmd.arg(flake.path.as_os_str()),
                };
            }
        }

        if let Some(max_jobs) = self.max_jobs {
            cmd.arg("--max-jobs").arg(max_jobs.to_string());
        }

        if let Some(cores) = self.cores {
            cmd.arg("--cores").arg(cores.to_string());
        }

        if self.print_build_logs {
            cmd.arg("-L");
        }

        let status = cmd.status()?;

        if !status.success() {
            eyre::bail!("nix exited with status {}", status);
        }

        Ok(())
    }
}
