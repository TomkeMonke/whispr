//! User settings, persisted as JSON in the app data directory.
//!
//! The Groq API key is deliberately not here - it lives in the OS credential
//! store. See `secrets.rs`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{groq, hotkey, local};

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
    /// Global shortcut, in the plugin's format, e.g. "Ctrl+Shift+Space".
    pub hotkey: String,
    /// Paste into the focused window. Off leaves the text on the clipboard.
    pub auto_paste: bool,
    /// Tidy the transcript with an LLM before it is pasted. Needs the Groq key.
    pub cleanup: bool,
    /// Names and jargon to spell right: a hint to Whisper, and a rule for cleanup.
    pub vocabulary: Vec<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            microphone: None,
            engine: Engine::Cloud,
            model: groq::MODEL_TURBO.to_string(),
            local_model: local::default_model().to_string(),
            language: "en".to_string(),
            hotkey: hotkey::DEFAULT.to_string(),
            auto_paste: true,
            cleanup: true,
            vocabulary: Vec::new(),
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
        self.vocabulary = clean_vocabulary(&self.vocabulary);
        self.hotkey = self.hotkey.trim().to_string();
        if hotkey::parse(&self.hotkey).is_none() {
            self.hotkey = hotkey::DEFAULT.to_string();
        }
        self
    }
}

/// Most terms a vocabulary keeps, and the longest term. Both bound the prompt
/// sizes: Whisper's hint is capped at 224 tokens.
const MAX_TERMS: usize = 100;
const MAX_TERM_CHARS: usize = 60;

/// Trim, drop blanks and over-long entries, and de-duplicate ignoring case,
/// keeping the first spelling given.
fn clean_vocabulary(terms: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    terms
        .iter()
        .map(|t| t.trim())
        .filter(|t| !t.is_empty() && t.chars().count() <= MAX_TERM_CHARS)
        .filter(|t| seen.insert(t.to_lowercase()))
        .take(MAX_TERMS)
        .map(str::to_string)
        .collect()
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
        assert_eq!(loaded.hotkey, hotkey::DEFAULT);
        assert!(loaded.auto_paste);
        assert!(loaded.cleanup);
        assert!(loaded.vocabulary.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unparseable_hotkey_falls_back() {
        let s = Settings {
            hotkey: "banana".into(),
            ..Default::default()
        }
        .sanitised();
        assert_eq!(s.hotkey, hotkey::DEFAULT);
    }

    #[test]
    fn vocabulary_is_tidied() {
        let s = Settings {
            vocabulary: vec![
                " drillr ".into(),
                "".into(),
                "Drillr".into(),
                "PostHog".into(),
                "x".repeat(200),
            ],
            ..Default::default()
        }
        .sanitised();
        assert_eq!(s.vocabulary, vec!["drillr", "PostHog"]);
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
            hotkey: "Ctrl+Alt+D".into(),
            auto_paste: false,
            cleanup: false,
            vocabulary: vec!["Tauri".into()],
        };
        save(&dir, &original).unwrap();
        let loaded = load(&dir);
        assert_eq!(loaded.microphone.as_deref(), Some("Yeti"));
        assert_eq!(loaded.model, groq::MODEL_LARGE);
        assert_eq!(loaded.engine, Engine::Local);
        assert_eq!(loaded.local_model, "base.en");
        assert_eq!(loaded.hotkey, "Ctrl+Alt+D");
        assert!(!loaded.auto_paste);
        assert!(!loaded.cleanup);
        assert_eq!(loaded.vocabulary, vec!["Tauri"]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
