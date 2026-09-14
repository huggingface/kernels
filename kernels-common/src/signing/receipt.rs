use serde::{Deserialize, Serialize};

use crate::git::Oid;

/// Kernel location.
///
/// Every variant must change when the kernel changes, e.g. through
/// an update. For a remote kernel, this is determined by the revision,
/// for local kernels this could e.g. be based on the file
/// names/sizes/mtimes.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KernelLocation {
    RemoteKernel {
        repo_id: String,
        revision: Oid,
        variant: String,
    },
}

impl KernelLocation {
    /// A Hub kernel.
    pub fn remote(repo_id: impl Into<String>, revision: Oid, variant: impl Into<String>) -> Self {
        KernelLocation::RemoteKernel {
            repo_id: repo_id.into(),
            revision,
            variant: variant.into(),
        }
    }
}
