use std::fs;
use std::io::{self, Write as _};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::git::Oid;
use crate::hf::{UnknownCacheDir, kernels_cache};

/// Directory in the kernels cache that holds all receipt stores.
const RECEIPTS_DIR: &str = ".verified-kernels";

/// Version of the on-disk signature receipt format.
pub const SIGNATURE_RECEIPT_FORMAT_VERSION: &str = "v1";

/// Version of the on-disk digest receipt format.
pub const DIGEST_RECEIPT_FORMAT_VERSION: &str = "v1";

/// Receipt of a successful verification of a kernel.
pub trait Receipt: Serialize + DeserializeOwned {
    /// The kernel location the verification applies to.
    fn location(&self) -> &KernelLocation;
}

/// Receipt of a successful signature verification of a kernel.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SignatureReceipt {
    /// The kernel location the receipt applies to.
    location: KernelLocation,
}

impl SignatureReceipt {
    pub fn new(location: KernelLocation) -> Self {
        SignatureReceipt { location }
    }
}

impl Receipt for SignatureReceipt {
    fn location(&self) -> &KernelLocation {
        &self.location
    }
}

/// Receipt of a successful digest verification of a kernel.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DigestReceipt {
    /// The kernel location the receipt applies to.
    location: KernelLocation,
}

impl DigestReceipt {
    pub fn new(location: KernelLocation) -> Self {
        DigestReceipt { location }
    }
}

impl Receipt for DigestReceipt {
    fn location(&self) -> &KernelLocation {
        &self.location
    }
}

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

    fn receipt_key(&self) -> String {
        let mut hasher = Sha256::new();
        match self {
            KernelLocation::RemoteKernel {
                repo_id,
                revision,
                variant,
            } => {
                update_with_string(&mut hasher, "remote_kernel");
                update_with_string(&mut hasher, repo_id);
                update_with_string(&mut hasher, revision.as_str());
                update_with_string(&mut hasher, variant);
            }
        }
        hex_encode(&hasher.finalize())
    }
}

/// Storage for verification receipts of type `R`.
///
/// Receipts are stored as JSON files in a cache directory, named by their
/// receipt key.
#[derive(Clone, Debug)]
pub struct ReceiptStore<R> {
    dir: PathBuf,
    receipt: PhantomData<R>,
}

/// Store for signature verification receipts.
pub type SignatureReceiptStore = ReceiptStore<SignatureReceipt>;

/// Store for digest verification receipts.
pub type DigestReceiptStore = ReceiptStore<DigestReceipt>;

impl SignatureReceiptStore {
    /// The signature receipt store inside the kernels cache.
    pub fn in_kernels_cache() -> Result<Self, ReceiptStoreError> {
        Ok(Self::in_cache_dir(&kernels_cache()?))
    }

    fn in_cache_dir(cache_dir: &Path) -> Self {
        Self::from_path(
            cache_dir
                .join(RECEIPTS_DIR)
                .join("signature")
                .join(SIGNATURE_RECEIPT_FORMAT_VERSION),
        )
    }
}

impl DigestReceiptStore {
    /// The digest receipt store inside the kernels cache.
    pub fn in_kernels_cache() -> Result<Self, ReceiptStoreError> {
        Ok(Self::in_cache_dir(&kernels_cache()?))
    }

    fn in_cache_dir(cache_dir: &Path) -> Self {
        Self::from_path(
            cache_dir
                .join(RECEIPTS_DIR)
                .join("digest")
                .join(DIGEST_RECEIPT_FORMAT_VERSION),
        )
    }
}

impl<R: Receipt> ReceiptStore<R> {
    /// A receipt store in the given directory.
    pub fn from_path(dir: impl Into<PathBuf>) -> Self {
        ReceiptStore {
            dir: dir.into(),
            receipt: PhantomData,
        }
    }

    /// Load the receipt for the given kernel location.
    ///
    /// Returns `Ok(None)` when no receipt exists for the location, and an
    /// error when a receipt exists but cannot be read, is corrupt, or
    /// describes a different kernel.
    pub fn load(&self, location: &KernelLocation) -> Result<Option<R>, ReceiptStoreError> {
        let path = self.dir.join(location.receipt_key());
        let data = match fs::read(&path) {
            Ok(data) => data,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(ReceiptStoreError::Read { path, source }),
        };
        let receipt: R =
            serde_json::from_slice(&data).map_err(|source| ReceiptStoreError::Corrupt {
                path: path.clone(),
                source,
            })?;

        if receipt.location() != location {
            return Err(ReceiptStoreError::LocationMismatch {
                path,
                found: Box::new(receipt.location().clone()),
            });
        }

        Ok(Some(receipt))
    }

    /// Store a receipt.
    pub fn store(&self, receipt: &R) -> Result<(), ReceiptStoreError> {
        let path = self.dir.join(receipt.location().receipt_key());
        let write_err = |source: io::Error| ReceiptStoreError::Write {
            path: path.clone(),
            source,
        };

        // This extra ceremony is so that we write the receipt atomically.
        let payload = serde_json::to_string(receipt).map_err(|e| write_err(io::Error::other(e)))?;
        fs::create_dir_all(&self.dir).map_err(&write_err)?;
        let mut tmp_file = tempfile::NamedTempFile::new_in(&self.dir).map_err(&write_err)?;
        tmp_file.write_all(payload.as_bytes()).map_err(&write_err)?;
        // Without this a crash can leave a zero-length receipt behind, which
        // reads back as `Corrupt` rather than as a plain cache miss.
        tmp_file.as_file().sync_all().map_err(&write_err)?;
        tmp_file.persist(&path).map_err(|e| write_err(e.error))?;
        Ok(())
    }
}

/// Error handling a verification receipt.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReceiptStoreError {
    /// The receipt file exists but cannot be read.
    #[error("cannot read receipt `{path}`")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    /// The receipt cannot be interpreted: invalid JSON, schema mismatch, or
    /// bad base64.
    #[error("receipt `{path}` is corrupt")]
    Corrupt {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// The receipt describes a different kernel than the one it was looked
    /// up for.
    #[error("receipt `{path}` describes a different kernel: {found:?}")]
    LocationMismatch {
        path: PathBuf,
        /// The location recorded in the receipt.
        found: Box<KernelLocation>,
    },

    /// The receipt cannot be written, e.g. because the cache is not
    /// writable. Also covers serialization failures.
    #[error("cannot store receipt `{path}`")]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    /// The receipt store location could not be determined.
    #[error(transparent)]
    UnknownCacheDir(#[from] UnknownCacheDir),
}

/// Length-prefixed string hash.
fn update_with_string(hasher: &mut impl Digest, part: &str) {
    hasher.update((part.len() as u64).to_be_bytes());
    hasher.update(part.as_bytes());
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(2 * bytes.len()), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use std::fmt::Debug;

    use super::*;
    use tempfile::TempDir;

    fn oid(hex_digit: &str) -> Oid {
        Oid::from_str(&hex_digit.repeat(40)).unwrap()
    }

    fn hub_location() -> KernelLocation {
        KernelLocation::remote(
            "kernels-test/signatures",
            oid("a"),
            "torch30-cxx11-cu128-x86_64-linux",
        )
    }

    #[test]
    fn receipt_key_is_deterministic_and_distinguishes_kernels() {
        let location = hub_location();
        assert_eq!(location.receipt_key(), location.receipt_key());

        for other in [
            KernelLocation::remote(
                "kernels-test/other",
                oid("a"),
                "torch30-cxx11-cu128-x86_64-linux",
            ),
            KernelLocation::remote(
                "kernels-test/signatures",
                oid("b"),
                "torch30-cxx11-cu128-x86_64-linux",
            ),
            KernelLocation::remote(
                "kernels-test/signatures",
                oid("a"),
                "torch30-cxx11-cpu-x86_64-linux",
            ),
        ] {
            assert_ne!(location.receipt_key(), other.receipt_key());
        }
    }

    /// The parts of a location are length-prefixed, so that moving a
    /// character from one part to the next cannot produce the same key.
    #[test]
    fn receipt_key_does_not_confuse_adjacent_parts() {
        let split_one = KernelLocation::remote("kernels-test/sig", oid("a"), "natures");
        let split_other = KernelLocation::remote("kernels-test/signatures", oid("a"), "");
        assert_ne!(split_one.receipt_key(), split_other.receipt_key());
    }

    #[test]
    fn receipt_roundtrip() {
        check_receipt_roundtrip(SignatureReceipt::new);
        check_receipt_roundtrip(DigestReceipt::new);
    }

    fn check_receipt_roundtrip<R: Receipt + Debug + PartialEq>(new: fn(KernelLocation) -> R) {
        let dir = TempDir::new().unwrap();
        let store = ReceiptStore::<R>::from_path(dir.path());
        let location = hub_location();
        let receipt = new(location.clone());

        store.store(&receipt).expect("receipt should store");
        let loaded = store
            .load(&location)
            .expect("receipt should load")
            .expect("receipt should exist");

        assert_eq!(loaded, receipt);
    }

    #[test]
    fn load_receipt_missing_or_corrupt() {
        check_load_receipt_missing_or_corrupt::<SignatureReceipt>();
        check_load_receipt_missing_or_corrupt::<DigestReceipt>();
    }

    fn check_load_receipt_missing_or_corrupt<R: Receipt + Debug>() {
        let dir = TempDir::new().unwrap();
        let store = ReceiptStore::<R>::from_path(dir.path());
        let location = hub_location();
        let key = location.receipt_key();

        assert!(matches!(store.load(&location), Ok(None)));

        fs::write(dir.path().join(&key), "not a receipt").unwrap();
        assert!(matches!(
            store.load(&location),
            Err(ReceiptStoreError::Corrupt { .. })
        ));

        // Valid JSON, but the location does not match the schema.
        fs::write(
            dir.path().join(&key),
            r#"{"location":{"type":"remote_kernel","repo_id":"kernels-test/signatures"}}"#,
        )
        .unwrap();
        assert!(matches!(
            store.load(&location),
            Err(ReceiptStoreError::Corrupt { .. })
        ));

        // A location variant that this version does not know. A newer
        // `kernels` may add one, and reading it must fail cleanly rather
        // than be misinterpreted: the caller then re-verifies and overwrites.
        fs::write(
            dir.path().join(&key),
            r#"{"location":{"type":"lunar_kernel","crater":"Tycho"}}"#,
        )
        .unwrap();
        assert!(matches!(
            store.load(&location),
            Err(ReceiptStoreError::Corrupt { .. })
        ));
    }

    #[test]
    fn load_receipt_rejects_transplanted_receipt() {
        check_load_receipt_rejects_transplanted_receipt(SignatureReceipt::new);
        check_load_receipt_rejects_transplanted_receipt(DigestReceipt::new);
    }

    fn check_load_receipt_rejects_transplanted_receipt<R: Receipt + Debug>(
        new: fn(KernelLocation) -> R,
    ) {
        let dir = TempDir::new().unwrap();
        let store = ReceiptStore::<R>::from_path(dir.path());

        let signed = hub_location();
        let unsigned = KernelLocation::remote(
            "kernels-test/signatures",
            oid("b"),
            "torch30-cxx11-cu128-x86_64-linux",
        );

        store
            .store(&new(signed.clone()))
            .expect("receipt should store");

        // Transplant the receipt onto the other revision's key.
        fs::copy(
            dir.path().join(signed.receipt_key()),
            dir.path().join(unsigned.receipt_key()),
        )
        .unwrap();

        match store.load(&unsigned) {
            Err(ReceiptStoreError::LocationMismatch { found, .. }) => {
                assert_eq!(*found, signed);
            }
            other => panic!("expected a location mismatch, got: {other:?}"),
        }

        // The receipt is still valid under its own key.
        assert!(store.load(&signed).unwrap().is_some());
    }

    #[test]
    fn store_receipt_fails_when_cache_not_writable() {
        check_store_receipt_fails_when_cache_not_writable(SignatureReceipt::new);
        check_store_receipt_fails_when_cache_not_writable(DigestReceipt::new);
    }

    fn check_store_receipt_fails_when_cache_not_writable<R: Receipt>(new: fn(KernelLocation) -> R) {
        let dir = TempDir::new().unwrap();
        let receipt_dir = dir.path().join("receipts");
        // A regular file where the receipt directory should be.
        fs::write(&receipt_dir, "not a directory").unwrap();

        let receipt = new(hub_location());
        assert!(matches!(
            ReceiptStore::<R>::from_path(&receipt_dir).store(&receipt),
            Err(ReceiptStoreError::Write { .. })
        ));
    }

    #[test]
    fn stores_use_separate_directories_in_cache() {
        let cache_dir = Path::new("/cache");

        assert_eq!(
            SignatureReceiptStore::in_cache_dir(cache_dir).dir,
            Path::new("/cache/.verified-kernels/signature/v1")
        );
        assert_eq!(
            DigestReceiptStore::in_cache_dir(cache_dir).dir,
            Path::new("/cache/.verified-kernels/digest/v1")
        );
    }

    /// The on-disk format must not change by accident. Changing it requires
    /// bumping the format version of the store.
    #[test]
    fn receipt_json_format_is_stable() {
        check_receipt_json_format_is_stable(SignatureReceipt::new);
        check_receipt_json_format_is_stable(DigestReceipt::new);
    }

    fn check_receipt_json_format_is_stable<R: Receipt + Debug + PartialEq>(
        new: fn(KernelLocation) -> R,
    ) {
        let receipt = new(hub_location());

        let json = serde_json::to_string(&receipt).unwrap();
        let expected = format!(
            r#"{{"location":{{"type":"remote_kernel","repo_id":"kernels-test/signatures","revision":"{}","variant":"torch30-cxx11-cu128-x86_64-linux"}}}}"#,
            "a".repeat(40)
        );
        assert_eq!(json, expected);

        // The pinned format must roundtrip.
        let parsed: R = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, receipt);
    }

    #[test]
    fn unknown_receipt_fields_are_ignored() {
        check_unknown_receipt_fields_are_ignored::<SignatureReceipt>();
        check_unknown_receipt_fields_are_ignored::<DigestReceipt>();
    }

    fn check_unknown_receipt_fields_are_ignored<R: Receipt + Debug>() {
        let dir = TempDir::new().unwrap();
        let store = ReceiptStore::<R>::from_path(dir.path());
        let location = hub_location();

        fs::write(
            dir.path().join(location.receipt_key()),
            format!(
                r#"{{"location":{{"type":"remote_kernel","repo_id":"kernels-test/signatures","revision":"{}","variant":"torch30-cxx11-cu128-x86_64-linux","from_the_future":[1,2,3]}},"verified_at":"2026-09-10T13:42:47Z"}}"#,
                "a".repeat(40)
            ),
        )
        .unwrap();

        let loaded = store
            .load(&location)
            .expect("receipt should load")
            .expect("receipt should exist");
        assert_eq!(loaded.location(), &location);
    }
}
