//! Groq API key storage.
//!
//! The key lives in the OS credential store (Windows Credential Manager /
//! macOS Keychain), never in a config file on disk.

const SERVICE: &str = "whispr";
const USER: &str = "groq-api-key";

#[derive(Debug, thiserror::Error)]
pub enum SecretError {
    #[error("credential store unavailable: {0}")]
    Store(#[from] keyring_core::Error),
}

/// Register the platform credential store. Must run once at startup, before any
/// entry is created.
pub fn init() -> Result<(), SecretError> {
    #[cfg(windows)]
    {
        keyring_core::set_default_store(windows_native_keyring_store::Store::new()?);
    }
    #[cfg(target_os = "macos")]
    {
        keyring_core::set_default_store(apple_native_keyring_store::Store::new()?);
    }
    Ok(())
}

/// The stored key, or `None` if one was never saved.
pub fn get() -> Result<Option<String>, SecretError> {
    let entry = keyring_core::Entry::new(SERVICE, USER)?;
    match entry.get_password() {
        Ok(secret) => {
            let trimmed = secret.trim().to_string();
            Ok(if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            })
        }
        Err(keyring_core::Error::NoEntry) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn set(key: &str) -> Result<(), SecretError> {
    let entry = keyring_core::Entry::new(SERVICE, USER)?;
    entry.set_password(key.trim())?;
    Ok(())
}

pub fn clear() -> Result<(), SecretError> {
    let entry = keyring_core::Entry::new(SERVICE, USER)?;
    match entry.delete_credential() {
        Ok(()) | Err(keyring_core::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.into()),
    }
}
