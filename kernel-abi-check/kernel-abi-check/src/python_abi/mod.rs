use std::collections::{BTreeSet, HashMap};

use eyre::Result;
use object::{BinaryFormat, ObjectSymbol, Symbol};
use once_cell::sync::Lazy;
use serde::Deserialize;

use crate::version::Version;

static ABI_TOML: &str = include_str!("stable_abi.toml");

#[derive(Deserialize)]
struct AbiInfoSerde {
    added: Version,
}

#[derive(Deserialize)]
struct StableAbiSerde {
    function: HashMap<String, AbiInfoSerde>,
    data: HashMap<String, AbiInfoSerde>,
}

#[derive(Clone, Copy, Debug)]
pub enum SymbolType {
    Data,
    Function,
}

#[derive(Clone, Debug)]
pub struct AbiInfo {
    #[allow(dead_code)]
    pub symbol_type: SymbolType,
    pub added: Version,
}

pub static PYTHON_STABLE_ABI: Lazy<HashMap<String, AbiInfo>> = Lazy::new(|| {
    let deserialized: StableAbiSerde = toml::de::from_str(ABI_TOML).unwrap();
    let mut symbols = HashMap::new();
    for (name, abi) in deserialized.function {
        symbols.insert(
            name,
            AbiInfo {
                symbol_type: SymbolType::Function,
                added: abi.added,
            },
        );
    }
    for (name, abi) in deserialized.data {
        symbols.insert(
            name,
            AbiInfo {
                symbol_type: SymbolType::Data,
                added: abi.added,
            },
        );
    }
    symbols
});

/// Python ABI violation.
#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub enum PythonAbiViolation {
    /// Symbol is newer than the specified Python ABI version.
    IncompatibleAbi3Symbol { name: String, added: Version },

    /// Symbol is not part of ABI3.
    NonAbi3Symbol { name: String },
}

/// Check for violations of the Python ABI policy.
pub fn check_python_abi<'a>(
    python_abi: &Version,
    binary_format: BinaryFormat,
    symbols: impl IntoIterator<Item = Symbol<'a, 'a>>,
) -> Result<BTreeSet<PythonAbiViolation>> {
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

        match PYTHON_STABLE_ABI.get(symbol_name) {
            Some(abi_info) => {
                if &abi_info.added > python_abi {
                    violations.insert(PythonAbiViolation::IncompatibleAbi3Symbol {
                        name: symbol_name.to_string(),
                        added: abi_info.added.clone(),
                    });
                }
            }
            None => {
                if symbol_name.starts_with("Py") || symbol_name.starts_with("_Py") {
                    violations.insert(PythonAbiViolation::NonAbi3Symbol {
                        name: symbol_name.to_string(),
                    });
                }
            }
        }
    }

    Ok(violations)
}
