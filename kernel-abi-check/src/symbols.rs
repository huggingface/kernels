use eyre::{bail, Result};
use object::read::elf::{ElfFile, FileHeader};
use object::{BinaryFormat, File, Object, ObjectSymbol};

/// A symbol that an object imports (an undefined symbol).
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct UndefinedSymbol {
    /// The symbol name.
    pub name: String,

    /// ELF symbol version, e.g. `GLIBC_2.14`. `None` for unversioned symbols.
    pub version: Option<String>,
}

/// Get the symbols that an object imports.
///
/// `data` must be the data that `file` was parsed from.
pub fn undefined_symbols(file: &File, data: &[u8]) -> Result<Vec<UndefinedSymbol>> {
    match file {
        File::Elf32(elf) => elf_undefined_symbols(elf, data),
        File::Elf64(elf) => elf_undefined_symbols(elf, data),
        File::MachO32(_) | File::MachO64(_) => macho_undefined_symbols(file),
        _ => bail!("Unsupported file format: {:?}", file.format()),
    }
}

fn elf_undefined_symbols<'data, Elf: FileHeader>(
    file: &ElfFile<'data, Elf>,
    data: &'data [u8],
) -> Result<Vec<UndefinedSymbol>> {
    let endian = file.endian();
    let versions = file.elf_section_table().versions(endian, data)?;

    let mut symbols = Vec::new();
    for symbol in file.dynamic_symbols() {
        if !symbol.is_undefined() {
            continue;
        }

        let version = match &versions {
            Some(versions) => versions
                .version(versions.version_index(endian, symbol.index()))?
                .map(|version| String::from_utf8_lossy(version.name()).into_owned()),
            None => None,
        };

        symbols.push(UndefinedSymbol {
            name: symbol.name()?.to_owned(),
            version,
        });
    }

    Ok(symbols)
}

fn macho_undefined_symbols(file: &File) -> Result<Vec<UndefinedSymbol>> {
    debug_assert!(matches!(file.format(), BinaryFormat::MachO));

    let mut symbols = Vec::new();
    for symbol in file.symbols() {
        if !symbol.is_undefined() {
            continue;
        }

        let name = symbol.name()?;
        symbols.push(UndefinedSymbol {
            // Mach-O C symbol mangling adds an underscore.
            name: name.strip_prefix('_').unwrap_or(name).to_owned(),
            version: None,
        });
    }

    Ok(symbols)
}
