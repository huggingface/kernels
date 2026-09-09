use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use thiserror::Error;
use x509_cert::Certificate;

use crate::cache;
use crate::variant::variant_files;

/// Version of the on-disk receipt format.
pub const CACHE_FORMAT_VERSION: &str = "v1";

/// Receipt of a successful kernel verification.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VerificationReceipt {
    /// Certificate from the sigstore bundle. The user-defined policy is
    /// verified against the certificate.
    #[serde(with = "certificate_base64")]
    certificate: Certificate,

    /// The kernel location the receipt applies to.
    location: KernelLocation,
}

impl VerificationReceipt {
    pub fn new(location: KernelLocation, certificate: Certificate) -> Self {
        VerificationReceipt {
            certificate,
            location,
        }
    }

    /// The signing certificate from the sigstore bundle.
    pub fn certificate(&self) -> &Certificate {
        &self.certificate
    }

    /// The kernel location the verification applies to.
    pub fn location(&self) -> &KernelLocation {
        &self.location
    }
}

/// Kernel location.
///
/// We need to invalidate the verification when was e.g. updated. This is
/// easy for remote kernels, since they have a unique revision. However,
/// for local kernels we need to compute a unique fingerprint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KernelLocation {
    LocalKernel {
        /// Canonicalized path of the variant directory.
        variant_path: PathBuf,
        /// Fingerprint of the variant's file metadata at verification time.
        #[serde(with = "base64_bytes")]
        fingerprint: Vec<u8>,
    },
    RemoteKernel {
        repo_id: String,
        revision: String,
        variant: String,
    },
}

impl KernelLocation {
    /// A local kernel.
    pub fn local(variant_path: impl Into<PathBuf>) -> io::Result<Self> {
        let variant_path = variant_path.into().canonicalize()?;
        let fingerprint = fingerprint_files(&variant_path)?;
        Ok(KernelLocation::LocalKernel {
            variant_path,
            fingerprint,
        })
    }

    /// A Hub kernel.
    pub fn remote(
        repo_id: impl Into<String>,
        revision: impl Into<String>,
        variant: impl Into<String>,
    ) -> Self {
        KernelLocation::RemoteKernel {
            repo_id: repo_id.into(),
            revision: revision.into(),
            variant: variant.into(),
        }
    }

    fn receipt_key(&self) -> String {
        let mut hasher = Sha256::new();
        match self {
            KernelLocation::LocalKernel {
                variant_path,
                fingerprint,
            } => {
                update(&mut hasher, "local");
                update(&mut hasher, &variant_path.to_string_lossy());
                hasher.update(fingerprint);
            }
            KernelLocation::RemoteKernel {
                repo_id,
                revision,
                variant,
            } => {
                update(&mut hasher, "hub");
                update(&mut hasher, repo_id);
                update(&mut hasher, revision);
                update(&mut hasher, variant);
            }
        }
        hex_encode(&hasher.finalize())
    }
}

/// Storage for verification receipts.
///
/// Receipts are stored as JSON files in a cache directory, named by their
/// receipt key.
#[derive(Clone, Debug)]
pub struct ReceiptStore {
    dir: PathBuf,
}

impl ReceiptStore {
    /// The receipt store at the default location in the kernel cache.
    pub fn new() -> Option<Self> {
        Some(Self::from_path(default_dir()?))
    }

    /// A receipt store in the given directory.
    pub fn from_path(dir: impl Into<PathBuf>) -> Self {
        ReceiptStore { dir: dir.into() }
    }

    /// Load the receipt for the given kernel location.
    ///
    /// Returns `Ok(None)` when no receipt exists for the location, and an
    /// error when a receipt exists but cannot be read, is corrupt, or
    /// describes a different kernel.
    pub fn load(
        &self,
        location: &KernelLocation,
    ) -> Result<Option<VerificationReceipt>, ReceiptStoreError> {
        let path = self.dir.join(location.receipt_key());
        let data = match fs::read(&path) {
            Ok(data) => data,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(ReceiptStoreError::Read { path, source }),
        };
        let receipt: VerificationReceipt =
            serde_json::from_slice(&data).map_err(|source| ReceiptStoreError::Corrupt {
                path: path.clone(),
                source,
            })?;

        if receipt.location != *location {
            return Err(ReceiptStoreError::LocationMismatch {
                path,
                found: Box::new(receipt.location),
            });
        }

        Ok(Some(receipt))
    }

    /// Store a receipt.
    pub fn store(&self, receipt: &VerificationReceipt) -> Result<(), ReceiptStoreError> {
        let path = self.dir.join(receipt.location.receipt_key());
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

    /// The receipt cannot be interpreted: invalid JSON, schema mismatch,
    /// bad base64, or an unparseable certificate.
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
}

/// The default receipt cache directory, inside the kernel cache.
fn default_dir() -> Option<PathBuf> {
    Some(
        cache::kernels_cache()?
            .join(".verified-kernel")
            .join(CACHE_FORMAT_VERSION),
    )
}

/// Length-prefixed string hash.
fn update(hasher: &mut Sha256, part: &str) {
    hasher.update((part.len() as u64).to_be_bytes());
    hasher.update(part.as_bytes());
}

/// Fingerprint the metadata of a local kernel.
///
/// We want to get cache misses when files have been modified, so use the
/// path, file size, and modification time. Local cache attacks are not
/// part of the attack vectors that we want to mitigate.
fn fingerprint_files(variant_path: &Path) -> io::Result<Vec<u8>> {
    let mut entries = Vec::new();
    for entry in variant_files(variant_path) {
        let (entry, metadata) = entry?;
        let relpath = entry
            .path()
            .strip_prefix(variant_path)
            .map_err(io::Error::other)?
            .to_string_lossy()
            .into_owned();
        entries.push((relpath, metadata.len(), mtime_ns(&metadata)?));
    }
    entries.sort();

    let mut hasher = Sha256::new();
    for (relpath, size, mtime_ns) in entries {
        update(&mut hasher, &relpath);
        hasher.update(size.to_be_bytes());
        hasher.update(mtime_ns.to_be_bytes());
    }
    Ok(hasher.finalize().to_vec())
}

/// Modification time in nanoseconds since the epoch (signed).
///
/// Saturates rather than wrapping for timestamps that do not fit, so that a
/// nonsensical mtime cannot alias a plausible one.
fn mtime_ns(metadata: &fs::Metadata) -> io::Result<i64> {
    let mtime = metadata.modified()?;
    Ok(match mtime.duration_since(UNIX_EPOCH) {
        Ok(duration) => i64::try_from(duration.as_nanos()).unwrap_or(i64::MAX),
        Err(e) => i64::try_from(e.duration().as_nanos()).map_or(i64::MIN, |nanos| -nanos),
    })
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

/// Serde adapter: base64-encoded bytes as a JSON string.
mod base64_bytes {
    use base64::prelude::{BASE64_STANDARD, Engine as _};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        BASE64_STANDARD.encode(bytes).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        let encoded = String::deserialize(deserializer)?;
        BASE64_STANDARD
            .decode(encoded.as_bytes())
            .map_err(serde::de::Error::custom)
    }
}

/// Serde adapter: certificate as base64-encoded DER. Deserialization parses
/// the certificate, so a corrupt certificate makes the whole receipt corrupt.
mod certificate_base64 {
    use base64::prelude::{BASE64_STANDARD, Engine as _};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use x509_cert::Certificate;
    use x509_cert::der::{Decode, Encode};

    pub fn serialize<S: Serializer>(cert: &Certificate, serializer: S) -> Result<S::Ok, S::Error> {
        let der = cert.to_der().map_err(serde::ser::Error::custom)?;
        BASE64_STANDARD.encode(der).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Certificate, D::Error> {
        let encoded = String::deserialize(deserializer)?;
        let der = BASE64_STANDARD
            .decode(encoded.as_bytes())
            .map_err(serde::de::Error::custom)?;
        Certificate::from_der(&der).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::prelude::{BASE64_STANDARD, Engine as _};
    use tempfile::TempDir;
    use x509_cert::der::{Decode, Encode};

    /// Self-signed P-256 certificate, generated with the same recipe as the
    /// Python `test_certificate` fixture.
    const TEST_CERTIFICATE: &str = "MIIBLTCB1KADAgECAhRNouRjUSo8HPRHW0H9/aZ3K3bcmjAKBggqhkjOPQQDAjAXMRUwEwYDVQQDDAxrZXJuZWxzLXRlc3QwHhcNMjYwOTA4MTEyOTUxWhcNMjYwOTEwMTEyOTUxWjAXMRUwEwYDVQQDDAxrZXJuZWxzLXRlc3QwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAAS849O3mGhg1f732geVdF9eSZlT/wM2VcYUsNUtPUNix5TuS5n9LvvbUy6X/xg8YQNJ33jln+K56w9bVLnsabGjMAoGCCqGSM49BAMCA0gAMEUCICygiEapSgOL43RWWDgwNMDc9EiiybkRmpxeUEhIFHDbAiEA5ibEqDAKcK9DoTYkmFAGJY0zwJjN6Mo/1OFstMqQMvI=";

    fn test_certificate() -> Certificate {
        let der = BASE64_STANDARD.decode(TEST_CERTIFICATE).unwrap();
        Certificate::from_der(&der).unwrap()
    }

    fn hub_location() -> KernelLocation {
        KernelLocation::remote(
            "kernels-test/signatures",
            "a".repeat(40),
            "torch30-cxx11-cu128-x86_64-linux",
        )
    }

    #[test]
    fn receipt_key_is_deterministic_and_distinguishes_origin() -> io::Result<()> {
        let dir = TempDir::new()?;
        let variant_path = dir.path().join("test-variant");
        fs::create_dir(&variant_path)?;

        let local = KernelLocation::local(&variant_path)?;
        let hub = hub_location();

        assert_eq!(local.receipt_key(), local.receipt_key());
        assert_eq!(hub.receipt_key(), hub.receipt_key());
        assert_ne!(local.receipt_key(), hub.receipt_key());
        Ok(())
    }

    #[test]
    fn local_receipt_key_changes_when_files_change() -> io::Result<()> {
        let dir = TempDir::new()?;
        let variant_path = dir.path().join("test-variant");
        fs::create_dir(&variant_path)?;
        fs::write(variant_path.join("kernel.py"), b"pass")?;

        let key_before = KernelLocation::local(&variant_path)?.receipt_key();

        // Different size, so the fingerprint changes even at equal mtime
        // granularity.
        fs::write(variant_path.join("kernel.py"), b"pass  # rebuilt")?;

        assert_ne!(
            key_before,
            KernelLocation::local(&variant_path)?.receipt_key()
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn local_receipt_key_follows_symlinks() -> io::Result<()> {
        let dir = TempDir::new()?;
        let blobs = dir.path().join("blobs");
        let variant_path = dir.path().join("snapshot");
        let empty_path = dir.path().join("empty");
        fs::create_dir(&blobs)?;
        fs::create_dir(&variant_path)?;
        fs::create_dir(&empty_path)?;

        fs::write(blobs.join("blob"), b"kernel code")?;
        std::os::unix::fs::symlink(blobs.join("blob"), variant_path.join("kernel.py"))?;

        let key_before = KernelLocation::local(&variant_path)?.receipt_key();

        // A symlinked file must contribute to the fingerprint at all.
        assert_ne!(
            key_before,
            KernelLocation::local(&empty_path)?.receipt_key()
        );

        // Rewriting the target must invalidate the receipt.
        fs::write(blobs.join("blob"), b"kernel code, but different")?;
        assert_ne!(
            key_before,
            KernelLocation::local(&variant_path)?.receipt_key()
        );
        Ok(())
    }

    #[test]
    fn local_receipt_key_ignores_python_bytecode() -> io::Result<()> {
        let dir = TempDir::new()?;
        let variant_path = dir.path().join("test-variant");
        fs::create_dir_all(variant_path.join("pkg"))?;
        fs::write(variant_path.join("pkg").join("kernel.py"), b"pass")?;

        let key_before = KernelLocation::local(&variant_path)?.receipt_key();

        let pycache = variant_path.join("pkg").join("__pycache__");
        fs::create_dir(&pycache)?;
        fs::write(pycache.join("kernel.cpython-311.pyc"), b"bytecode")?;
        fs::write(variant_path.join("top-level.pyc"), b"bytecode")?;

        assert_eq!(
            key_before,
            KernelLocation::local(&variant_path)?.receipt_key()
        );
        Ok(())
    }

    #[test]
    fn local_receipt_key_covers_metadata_and_signature() -> io::Result<()> {
        let dir = TempDir::new()?;
        let variant_path = dir.path().join("test-variant");
        fs::create_dir(&variant_path)?;
        fs::write(variant_path.join("kernel.py"), b"pass")?;
        fs::write(variant_path.join("metadata.json"), b"{}")?;
        fs::write(variant_path.join("metadata.json.sigstore"), b"sig")?;

        let key_before = KernelLocation::local(&variant_path)?.receipt_key();

        fs::write(variant_path.join("metadata.json.sigstore"), b"other sig")?;
        let key_resigned = KernelLocation::local(&variant_path)?.receipt_key();
        assert_ne!(key_before, key_resigned);

        fs::write(variant_path.join("metadata.json"), b"{\"a\": 1}")?;
        assert_ne!(
            key_resigned,
            KernelLocation::local(&variant_path)?.receipt_key()
        );
        Ok(())
    }

    #[test]
    fn receipt_roundtrip() -> io::Result<()> {
        let dir = TempDir::new()?;
        let store = ReceiptStore::from_path(dir.path());
        let location = hub_location();
        let receipt = VerificationReceipt::new(location.clone(), test_certificate());

        store.store(&receipt).expect("receipt should store");
        let loaded = store
            .load(&location)
            .expect("receipt should load")
            .expect("receipt should exist");

        assert_eq!(
            loaded.certificate().to_der().unwrap(),
            test_certificate().to_der().unwrap()
        );
        assert_eq!(loaded.location(), receipt.location());
        Ok(())
    }

    #[test]
    fn load_receipt_missing_or_corrupt() {
        let dir = TempDir::new().unwrap();
        let store = ReceiptStore::from_path(dir.path());
        let location = hub_location();
        let key = location.receipt_key();

        assert!(matches!(store.load(&location), Ok(None)));

        fs::write(dir.path().join(&key), "not a receipt").unwrap();
        assert!(matches!(
            store.load(&location),
            Err(ReceiptStoreError::Corrupt { .. })
        ));

        // Valid JSON with a corrupt certificate.
        fs::write(
            dir.path().join(&key),
            r#"{"certificate":"aGVsbG8=","location":{"type":"remote_kernel","repo_id":"kernels-test/signatures","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","variant":"torch30-cxx11-cu128-x86_64-linux"}}"#,
        )
        .unwrap();
        assert!(matches!(
            store.load(&location),
            Err(ReceiptStoreError::Corrupt { .. })
        ));
    }

    #[test]
    fn load_receipt_rejects_transplanted_receipt() {
        let dir = TempDir::new().unwrap();
        let store = ReceiptStore::from_path(dir.path());

        let signed = hub_location();
        let unsigned = KernelLocation::remote(
            "kernels-test/signatures",
            "b".repeat(40),
            "torch30-cxx11-cu128-x86_64-linux",
        );

        store
            .store(&VerificationReceipt::new(
                signed.clone(),
                test_certificate(),
            ))
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
        let dir = TempDir::new().unwrap();
        let receipt_dir = dir.path().join("receipts");
        // A regular file where the receipt directory should be.
        fs::write(&receipt_dir, "not a directory").unwrap();

        let receipt = VerificationReceipt::new(hub_location(), test_certificate());
        assert!(matches!(
            ReceiptStore::from_path(&receipt_dir).store(&receipt),
            Err(ReceiptStoreError::Write { .. })
        ));
    }

    /// Pins the v1 on-disk format: field names, enum tagging, base64 DER.
    #[test]
    fn receipt_json_format_is_stable() {
        let receipt = VerificationReceipt::new(hub_location(), test_certificate());

        let json = serde_json::to_string(&receipt).unwrap();
        let expected = format!(
            r#"{{"certificate":"{TEST_CERTIFICATE}","location":{{"type":"remote_kernel","repo_id":"kernels-test/signatures","revision":"{}","variant":"torch30-cxx11-cu128-x86_64-linux"}}}}"#,
            "a".repeat(40)
        );
        assert_eq!(json, expected);

        // The pinned format must roundtrip.
        let parsed: VerificationReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.location(), receipt.location());
    }
}
