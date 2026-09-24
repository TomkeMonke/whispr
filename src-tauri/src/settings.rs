//! User settings, persisted as JSON in the app data directory.
//!
//! The Groq API key is deliberately not here - it lives in the OS credential
//! store. See `secrets.rs`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{groq, local};

/// Which engine is tried first. The other one is the fallback, whenever it is
/// ready to go (a model on disk, or a key in the keychain).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    Cloud,
    Local,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Microphone name, or `None` to follow the system default.
    pub microphone: Option<String>,
    pub engine: Engine,
    /// Groq model.
    pub model: String,
    /// Local model id, from `local::MODELS`.
    pub local_model: String,
    /// Pinning the language skips Whisper's detection pass, which is both
    /// slightly faster and more accurate on short clips.
    pub language: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            microphone: None,
            engine: Engine::Cloud,
            model: groq::MODEL_TURBO.to_string(),
            local_model: local::default_model().to_string(),
            language: "en".to_string(),
        }
    }
}

impl Settings {
    /// Fall back to defaults for anything nonsensical, so a hand-edited or
    /// half-written file can never brick startup.
    pub fn sanitised(mut self) -> Self {
        if !groq::is_known_model(&self.model) {
            self.model = groq::MODEL_TURBO.to_string();
        }
        if local::find(&self.local_model).is_none() {
            self.local_model = local::default_model().to_string();
        }
        if self.language.trim().is_empty() {
            self.language = "en".to_string();
        }
        if self.microphone.as_deref().map(str::trim) == Some("") {
            self.microphone = None;
        }
        self
    }
}

pub fn path(dir: &Path) -> PathBuf {
    dir.join("settings.json")
}

/// Read settings, falling back to defaults if the file is missing or corrupt.
pub fn load(dir: &Path) -> Settings {
    match std::fs::read_to_string(path(dir)) {
        Ok(raw) => serde_json::from_str::<Settings>(&raw)
            .unwrap_or_default()
            .sanitised(),
        Err(_) => Settings::default(),
    }
}

pub fn save(dir: &Path, settings: &Settings) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let json = serde_json::to_string_pretty(settings)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path(dir), json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_usable() {
        let s = Settings::default();
        assert!(groq::is_known_model(&s.model));
        assert_eq!(s.language, "en");
        assert!(s.microphone.is_none());
    }

    #[test]
    fn unknown_model_falls_back() {
        let s = Settings {
            model: "whisper-tiny-imaginary".into(),
            ..Default::default()
        }
        .sanitised();
        assert_eq!(s.model, groq::MODEL_TURBO);
    }

    #[test]
    fn unknown_local_model_falls_back() {
        let s = Settings {
            local_model: "ggml-imaginary".into(),
            ..Default::default()
        }
        .sanitised();
        assert_eq!(s.local_model, local::default_model());
    }

    #[test]
    fn m1_settings_file_still_loads() {
        // Written before M3 added the engine fields.
        let dir = std::env::temp_dir().join(format!("whispr-m1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            path(&dir),
            r#"{"microphone":null,"model":"whisper-large-v3","language":"en"}"#,
        )
        .unwrap();
        let loaded = load(&dir);
        assert_eq!(loaded.model, groq::MODEL_LARGE);
        assert_eq!(loaded.engine, Engine::Cloud);
        assert_eq!(loaded.local_model, local::default_model());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn engine_serialises_lowercase() {
        assert_eq!(serde_json::to_string(&Engine::Local).unwrap(), r#""local""#);
    }

    #[test]
    fn blank_fields_fall_back() {
        let s = Settings {
            microphone: Some("   ".into()),
            language: "  ".into(),
            ..Default::default()
        }
        .sanitised();
        assert!(s.microphone.is_none());
        assert_eq!(s.language, "en");
    }

    #[test]
    fn corrupt_file_yields_defaults() {
        let dir = std::env::temp_dir().join(format!("whispr-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(path(&dir), "{ this is not json").unwrap();
        let loaded = load(&dir);
        assert_eq!(loaded.model, groq::MODEL_TURBO);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = std::env::temp_dir().join(format!("whispr-rt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let original = Settings {
            microphone: Some("Yeti".into()),
            engine: Engine::Local,
            model: groq::MODEL_LARGE.into(),
            local_model: "base.en".into(),
            language: "en".into(),
        };
        save(&dir, &original).unwrap();
        let loaded = load(&dir);
        assert_eq!(loaded.microphone.as_deref(), Some("Yeti"));
        assert_eq!(loaded.model, groq::MODEL_LARGE);
        assert_eq!(loaded.engine, Engine::Local);
        assert_eq!(loaded.local_model, "base.en");
        std::fs::remove_dir_all(&dir).ok();
    }
}
