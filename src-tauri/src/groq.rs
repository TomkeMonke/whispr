//! Groq speech-to-text client.
//!
//! The request is made from Rust rather than the webview so the API key never
//! crosses into JavaScript.

use std::time::Duration;

const API_URL: &str = "https://api.groq.com/openai/v1/audio/transcriptions";

/// Groq's production transcription models. `distil-whisper-large-v3-en` was
/// retired from their lineup, so turbo is the cheap fast default.
pub const MODEL_TURBO: &str = "whisper-large-v3-turbo";
pub const MODEL_LARGE: &str = "whisper-large-v3";

pub fn is_known_model(model: &str) -> bool {
    matches!(model, MODEL_TURBO | MODEL_LARGE)
}

#[derive(Debug, thiserror::Error)]
pub enum GroqError {
    #[error("no Groq API key saved - add one in settings")]
    MissingKey,
    #[error("Groq rejected the API key - check it in settings")]
    BadKey,
    #[error("Groq rate limit reached - try again shortly, or switch to local")]
    RateLimited,
    #[error("recording is too long for one request - keep it under about 13 minutes")]
    TooLarge,
    #[error("could not reach Groq - check your connection")]
    Offline,
    #[error("Groq returned {status}: {body}")]
    Api { status: u16, body: String },
    #[error("could not read Groq's reply: {0}")]
    Malformed(String),
}

impl GroqError {
    /// Whether falling back to the local engine is worth trying.
    /// A bad key or an oversized clip will fail the same way on a retry.
    pub fn should_fall_back(&self) -> bool {
        matches!(
            self,
            GroqError::Offline | GroqError::RateLimited | GroqError::Api { .. }
        )
    }
}

#[derive(serde::Deserialize)]
struct TranscriptionResponse {
    text: String,
}

pub fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .connect_timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_default()
}

/// Transcribe a 16 kHz mono WAV.
///
/// `prompt` is Whisper's context hint - roughly 224 tokens of names and jargon
/// that nudge spelling. The vocabulary from settings arrives here.
pub async fn transcribe(
    client: &reqwest::Client,
    api_key: &str,
    wav: Vec<u8>,
    model: &str,
    language: Option<&str>,
    prompt: Option<&str>,
) -> Result<String, GroqError> {
    if api_key.is_empty() {
        return Err(GroqError::MissingKey);
    }

    let part = reqwest::multipart::Part::bytes(wav)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| GroqError::Malformed(e.to_string()))?;

    let mut form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", model.to_string())
        .text("response_format", "json".to_string())
        // Deterministic: we want the same audio to give the same text.
        .text("temperature", "0".to_string());

    if let Some(lang) = language {
        form = form.text("language", lang.to_string());
    }
    if let Some(hint) = prompt {
        if !hint.trim().is_empty() {
            form = form.text("prompt", hint.trim().to_string());
        }
    }

    let response = client
        .post(API_URL)
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .await
        .map_err(|e| {
            if e.is_connect() || e.is_timeout() {
                GroqError::Offline
            } else {
                GroqError::Malformed(e.to_string())
            }
        })?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(match status.as_u16() {
            401 | 403 => GroqError::BadKey,
            413 => GroqError::TooLarge,
            429 => GroqError::RateLimited,
            other => GroqError::Api {
                status: other,
                body: truncate(&body, 400),
            },
        });
    }

    let parsed: TranscriptionResponse = response
        .json()
        .await
        .map_err(|e| GroqError::Malformed(e.to_string()))?;

    Ok(parsed.text)
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect::<String>() + "..."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bad_key_does_not_trigger_local_fallback() {
        assert!(!GroqError::BadKey.should_fall_back());
        assert!(!GroqError::MissingKey.should_fall_back());
        assert!(!GroqError::TooLarge.should_fall_back());
    }

    #[test]
    fn transport_failures_do_trigger_local_fallback() {
        assert!(GroqError::Offline.should_fall_back());
        assert!(GroqError::RateLimited.should_fall_back());
    }

    #[test]
    fn truncate_keeps_short_strings_intact() {
        assert_eq!(truncate("  hello  ", 400), "hello");
        assert_eq!(truncate("abcdef", 3), "abc...");
    }
}
