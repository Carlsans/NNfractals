//! Content-hash-keyed persistence for `anim_timeline::AnimationTimeline`.
//!
//! No sidecar-file convention existed anywhere in this codebase before this
//! (every `with_extension` use elsewhere is `.nn`<->`.png` preview images),
//! so this is new, additive infrastructure. Keyed by the genome's own
//! `Genome::content_hash()` (FNV-1a over program/warp/julia/phoenix/
//! bailout — same recipe `QuatIndividual::content_hash` already uses for
//! the MAP-Elites archive), NOT by file path, so renaming or copying a
//! `.nn` file never loses its animation settings.

use std::path::PathBuf;
use anyhow::Result;

use crate::anim_timeline::AnimationTimeline;

fn anim_settings_dir() -> PathBuf {
    crate::project_root().join("anim_settings")
}

/// Where `content_hash`'s timeline would be saved/loaded from.
pub fn timeline_path(content_hash: u64) -> PathBuf {
    anim_settings_dir().join(format!("{content_hash:016x}.json"))
}

/// Loads the saved timeline for `content_hash`, or `None` if this fractal
/// has never had one saved (or the file is unreadable/corrupt — treated the
/// same as "never saved" rather than a hard error, since the caller's
/// natural fallback is `AnimationTimeline::new(content_hash)` either way).
pub fn load_timeline(content_hash: u64) -> Option<AnimationTimeline> {
    let path = timeline_path(content_hash);
    let json = std::fs::read_to_string(&path).ok()?;
    let mut tl: AnimationTimeline = serde_json::from_str(&json).ok()?;
    tl.migrate_audio_track();
    Some(tl)
}

/// Saves `timeline` under its own `genome_content_hash`, matching
/// `io::save_genome`'s `serde_json::to_string_pretty` convention exactly.
/// Creates `anim_settings/` on first use.
pub fn save_timeline(timeline: &AnimationTimeline) -> Result<()> {
    let dir = anim_settings_dir();
    std::fs::create_dir_all(&dir)?;
    let path = timeline_path(timeline.genome_content_hash);
    let json = serde_json::to_string_pretty(timeline)?;
    std::fs::write(&path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_then_load_round_trips() {
        // A distinctive hash so this test's artifact can't collide with a
        // real genome's content hash, and is easy to spot/clean up under
        // anim_settings/ if the removal below is ever skipped by a panic.
        let hash: u64 = 0x0BADC0DE_0BADC0DE;
        let mut tl = AnimationTimeline::new(hash);
        tl.delta_t = 2.5;
        tl.duration_s = 12.0;

        save_timeline(&tl).expect("save should succeed");
        let loaded = load_timeline(hash).expect("the file we just saved should load back");
        assert_eq!(loaded, tl);

        // Clean up the test artifact so repeated `cargo test` runs don't
        // accumulate files.
        let _ = std::fs::remove_file(timeline_path(hash));
    }

    #[test]
    fn loading_a_hash_that_was_never_saved_is_none() {
        assert!(load_timeline(0x1111_2222_3333_4444).is_none());
    }

    #[test]
    fn loading_a_pre_migration_json_file_from_disk_migrates_audio() {
        // Writes a raw, hand-built JSON string directly to disk (bypassing
        // save_timeline, which would already write the NEW shape) shaped
        // like a file saved before AudioClip/audio_clips existed: only the
        // old "audio" key, no "audio_clips" key at all. Confirms the
        // migration actually fires at the real persistence boundary
        // (load_timeline), not just on an in-memory struct built by hand.
        let hash: u64 = 0x0BADC0DE_11111111;
        let path = timeline_path(hash);
        std::fs::create_dir_all(path.parent().unwrap()).expect("create anim_settings/");
        let old_shaped = serde_json::json!({
            "genome_content_hash": hash,
            "duration_s": 10.0, "trim_start_s": 0.0, "trim_end_s": 10.0,
            "audio": { "file_path": "/tmp/pre_migration.wav", "offset_s": 0.5, "gain_db": 0.0 },
        });
        std::fs::write(&path, serde_json::to_string_pretty(&old_shaped).unwrap()).expect("write raw old-shaped JSON");

        let loaded = load_timeline(hash).expect("should load and migrate");
        assert!(loaded.audio.is_none(), "deprecated field must be cleared by the time load_timeline returns");
        assert_eq!(loaded.audio_clips.len(), 1);
        assert_eq!(loaded.audio_clips[0].offset_s, 0.5);

        let _ = std::fs::remove_file(&path);
    }
}
