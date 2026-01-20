use std::path::PathBuf;
use std::{collections::BTreeSet, fs};

use clap::Parser;
use eyre::{Context, Result};
use object::{File, Object};

use kernel_abi_check::{
    check_macos, check_manylinux, check_python_abi, MacOSViolation, ManylinuxViolation,
    PythonAbiViolation, Version,
};

/// CLI tool to check library versions
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Python extension library.
    object: PathBuf,

    /// Manylinux version.
    #[arg(short, long, value_name = "VERSION", default_value = "manylinux_2_28")]
    manylinux: String,

    /// macOS version.
    #[arg(long, value_name = "VERSION", default_value = "15.0")]
    macos: Version,

    /// Python ABI version.
    #[arg(short, long, value_name = "VERSION", default_value = "3.9")]
    python_abi: Version,
}

fn main() -> Result<()> {
    // Initialize color_eyre error handling
    color_eyre::install()?;

    // Parse command-line arguments
    let args = Cli::parse();

    let binary_data = fs::read(args.object).context("Cannot open object file")?;
    let file = object::File::parse(&*binary_data).context("Cannot parse object")?;

    let mut manylinux_violations = BTreeSet::new();
    let mut macos_violations = BTreeSet::new();

    match file {
        File::Elf32(_) | File::Elf64(_) => {
            // Assume for now that ELF is Linux.
            eprintln!(
                "🐍 Checking for compatibility with {} and Python ABI version {}",
                args.manylinux, args.python_abi
            );

            manylinux_violations = check_manylinux(
                &args.manylinux,
                file.architecture(),
                file.endianness(),
                file.symbols(),
            )?;
            print_manylinux_violations(&manylinux_violations, &args.manylinux)?;
        }
        File::MachO32(_) | File::MachO64(_) => {
            eprintln!(
                "🐍 Checking for compatibility with macOS {}, and Python ABI version {}",
                args.macos, args.python_abi
            );
            macos_violations = check_macos(&file, &args.macos)?;
            print_macos_violations(&macos_violations, &args.macos);
        }
        _ => {
            return Err(eyre::eyre!("Unsupported file format: {:?}", file.format()));
        }
    }

    let python_abi_violations = check_python_abi(&args.python_abi, file.format(), file.symbols())?;
    print_python_abi_violations(&python_abi_violations, &args.python_abi);

    if !(manylinux_violations.is_empty()
        && macos_violations.is_empty()
        && python_abi_violations.is_empty())
    {
        return Err(eyre::eyre!("Compatibility issues found"));
    } else {
        eprintln!("✅ No compatibility issues found");
    }

    Ok(())
}

fn print_manylinux_violations(
    violations: &BTreeSet<ManylinuxViolation>,
    manylinux_version: &str,
) -> Result<()> {
    if !violations.is_empty() {
        eprintln!("\n⛔ Symbols incompatible with `{manylinux_version}` found:\n",);
        for violation in violations {
            match violation {
                ManylinuxViolation::Symbol { name, dep, version } => {
                    eprintln!("{name}_{dep}: {version}");
                }
            }
        }
    }
    Ok(())
}

fn print_macos_violations(violations: &BTreeSet<MacOSViolation>, macos_version: &Version) {
    if !violations.is_empty() {
        for violation in violations {
            match violation {
                MacOSViolation::MissingMinOS => {
                    eprintln!("\n⛔ shared library does not contain minimum macOS version");
                }
                MacOSViolation::IncompatibleMinOS { version } => {
                    eprintln!(
                        "\n⛔ shared library requires macOS version {version}, which is newer than {macos_version}",
                    );
                }
            }
        }
    }
}

fn print_python_abi_violations(violations: &BTreeSet<PythonAbiViolation>, python_abi: &Version) {
    if !violations.is_empty() {
        let newer_abi3_symbols = violations
            .iter()
            .filter(|v| matches!(v, PythonAbiViolation::IncompatibleAbi3Symbol { .. }))
            .collect::<BTreeSet<_>>();
        let non_abi3_symbols = violations
            .iter()
            .filter(|v| matches!(v, PythonAbiViolation::NonAbi3Symbol { .. }))
            .collect::<BTreeSet<_>>();

        if !newer_abi3_symbols.is_empty() {
            eprintln!("\n⛔ Symbols >= Python ABI {python_abi} found:\n",);
            for violation in newer_abi3_symbols {
                if let PythonAbiViolation::IncompatibleAbi3Symbol { name, added } = violation {
                    eprintln!("{name}: {added}");
                }
            }
        }

        if !non_abi3_symbols.is_empty() {
            eprintln!("\n⛔ Non-ABI3 symbols found:\n");
            for violation in &non_abi3_symbols {
                if let PythonAbiViolation::NonAbi3Symbol { name } = violation {
                    eprintln!("{name}");
                }
            }
        }
    }
}
