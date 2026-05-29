use std::collections::{BTreeSet, HashMap};
use std::str::FromStr;

use cpp_demangle::Symbol as CppSymbol;
use eyre::Result;
use object::{BinaryFormat, ObjectSymbol, Symbol};
use once_cell::sync::Lazy;

use crate::version::Version;

// https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/torch/csrc/stable/c/shim_function_versions.txt
static SHIM_FUNCTION_VERSIONS_RAW: &str = include_str!("shim_function_versions.txt");

/// Maps shim function names to the minimum Torch version that introduced them.
/// Functions absent from this map were available before 2.10.0.
pub static TORCH_SHIM_VERSIONS: Lazy<HashMap<String, Version>> = Lazy::new(|| {
    let mut map = HashMap::new();
    for line in SHIM_FUNCTION_VERSIONS_RAW.lines() {
        // Skip blank lines and comments.
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((name, version_token)) = line.split_once(':') {
            let name = name.trim().to_owned();
            // TORCH_VERSION_2_10_0 -> "2.10.0"
            let version_str = version_token
                .trim()
                .strip_prefix("TORCH_VERSION_")
                .expect("unexpected version token format")
                .replace('_', ".");
            let version = Version::from_str(&version_str)
                .expect("invalid version in shim_function_versions.txt");
            map.insert(name, version);
        }
    }
    map
});

/// Torch stable ABI violation.
#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub enum TorchStableAbiViolation {
    /// Symbol is newer than the specified Torch Stable ABI version.
    IncompatibleStableAbiSymbol { name: String, added: Version },

    /// Symbol is not part of ABI3.
    NonStableAbiSymbol { name: String },
}

/// Check for violations of the Python ABI policy.
pub fn check_torch_stable_abi<'a>(
    torch_stable_abi: &Version,
    binary_format: BinaryFormat,
    symbols: impl IntoIterator<Item = Symbol<'a, 'a>>,
) -> Result<BTreeSet<TorchStableAbiViolation>> {
    let mut violations = BTreeSet::new();

    for symbol in symbols {
        if !symbol.is_undefined() {
            continue;
        }

        let mut symbol_name = symbol.name()?;
        if matches!(binary_format, BinaryFormat::MachO) {
            // Mach-O C symbol mangling adds an underscore.
            symbol_name = symbol_name.strip_prefix("_").unwrap_or(symbol_name);
        }

        // If this is a C shim symbol, check if it is valid for this version.
        if let Some(symbol_version) = TORCH_SHIM_VERSIONS.get(symbol_name) {
            if symbol_version > torch_stable_abi {
                violations.insert(TorchStableAbiViolation::IncompatibleStableAbiSymbol {
                    name: symbol_name.to_owned(),
                    added: symbol_version.clone(),
                });
            }
            continue;
        }

        // Try to demangle the symbol as a C++ symbol. If that fails, it's probably an
        // unrelated C symbol.
        let cpp_symbol = match CppSymbol::new(symbol_name) {
            Ok(cpp_symbol) => cpp_symbol,
            Err(_) => {
                continue;
            }
        };
        let demangled = cpp_symbol.demangle()?;

        // Check if Torch symbols are from the stable ABI.
        if demangled.starts_with("torch::stable::") {
            // This branch fulfills to purposes: (1) avoid that stable ABI
            // C++ symbols get reported by the filter below. (2) Once a
            // versioned list of symbols is available, check versions.
        } else if demangled.starts_with("c10::")
            || demangled.starts_with("at::")
            || demangled.starts_with("torch::")
        {
            violations.insert(TorchStableAbiViolation::NonStableAbiSymbol { name: demangled });
        }
    }

    Ok(violations)
}
