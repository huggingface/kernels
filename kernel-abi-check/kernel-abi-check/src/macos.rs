use std::collections::BTreeSet;

use eyre::{Context, Result};
use object::macho::{BuildVersionCommand, LC_BUILD_VERSION};
use object::read::macho::{MachHeader, MachOFile};
use object::File;

use crate::Version;

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MacOSViolation {
    MissingMinOS,
    IncompatibleMinOS { version: Version },
}

fn build_version<Mach: MachHeader>(macho64: &MachOFile<Mach>) -> Result<Option<Version>> {
    let mut load_commands = macho64
        .macho_load_commands()
        .context("Cannot get Mach-O binary load commands")?;
    while let Some(load_command) = load_commands.next().context("Cannot get load command")? {
        if load_command.cmd() == LC_BUILD_VERSION {
            let command = load_command.data::<BuildVersionCommand<Mach::Endian>>()?;
            let version_u32 = command.minos.get(macho64.endian());
            let major = (version_u32 >> 16) as usize;
            let minor = ((version_u32 >> 8) as u8) as usize;
            let patch = (version_u32 as u8) as usize;

            return Ok(Some(Version::from(vec![major, minor, patch])));
        }
    }

    Ok(None)
}

/// Check that kernel binary is compatible with the given macOS version.
pub fn check_macos(file: &File, macos_version: &Version) -> Result<BTreeSet<MacOSViolation>> {
    let mut violations = BTreeSet::new();

    let minos = if let File::MachO64(macho64) = &file {
        build_version(macho64)?
    } else {
        return Ok(violations);
    };

    match minos {
        Some(object_version) => {
            if &object_version > macos_version {
                violations.insert(MacOSViolation::IncompatibleMinOS {
                    version: object_version,
                });
            }
        }
        None => {
            violations.insert(MacOSViolation::MissingMinOS);
        }
    }

    Ok(violations)
}
