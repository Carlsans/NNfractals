//! Named, genome-agnostic snapshots of an `AnimationTimeline` — "profiles"
//! — that Carl can save once and re-apply to any other fractal. Distinct
//! from `anim_persist`'s per-genome (content-hash-keyed) persistence: a
//! profile is keyed by a USER-CHOSEN NAME, not tied to any one genome, and
//! applying one to a fractal deliberately overwrites that fractal's own
//! current timeline settings (axis assignment, bounding box, every track,
//! effects lanes, audio clips, delta_t, trim, camera) — everything except
//! which genome it belongs to, which the caller must set explicitly after
//! loading (see `AnimProfile::timeline`'s own doc comment).

use std::path::PathBuf;
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::anim_timeline::AnimationTimeline;

fn profiles_dir() -> PathBuf {
    crate::project_root().join("anim_profiles")
}

/// Filesystem-safe stand-in for a profile's display name. Two names that
/// sanitize to the same string collide on disk (the second save overwrites
/// the first) — accepted as a rare, low-stakes edge case rather than adding
/// UUID-keyed storage for a purely cosmetic, user-facing name.
fn sanitize_name(name: &str) -> String {
    let s: String = name.trim().chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' { c } else { '_' })
        .collect();
    let s = s.trim().replace(' ', "_");
    if s.is_empty() { "profile".to_string() } else { s }
}

fn profile_path(name: &str) -> PathBuf {
    profiles_dir().join(format!("{}.json", sanitize_name(name)))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnimProfile {
    pub name: String,
    /// The saved timeline, minus any genome identity — `genome_content_hash`
    /// is always written as `0` (see `save_profile`) and MUST be overwritten
    /// with the target fractal's own hash by whoever applies this profile
    /// (`anim_viewer.rs::App::apply_profile`), since `anim_persist::
    /// save_timeline` uses that field as its own on-disk key — applying a
    /// profile without fixing it up would silently corrupt a DIFFERENT
    /// genome's saved settings the next time anything persists.
    pub timeline: AnimationTimeline,
}

/// Every saved profile's display name, sorted alphabetically
/// (case-insensitive) for a stable, predictable list in the UI. Corrupt or
/// unreadable files are silently skipped rather than failing the whole
/// listing — the same "treat unreadable as absent" stance
/// `anim_persist::load_timeline` takes.
pub fn list_profiles() -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(profiles_dir()) else { return Vec::new(); };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .filter_map(|s| serde_json::from_str::<AnimProfile>(&s).ok())
        .map(|p| p.name)
        .collect();
    names.sort_by_key(|n| n.to_lowercase());
    names
}

/// Saves `timeline` as a profile under `name`, creating `anim_profiles/` on
/// first use. Overwrites any existing profile with the same (sanitized)
/// name — this one function covers both "create a new profile" and "save
/// over an existing one," which differ only in whether the name was
/// already taken.
pub fn save_profile(name: &str, timeline: &AnimationTimeline) -> Result<()> {
    let dir = profiles_dir();
    std::fs::create_dir_all(&dir)?;
    let mut tl = timeline.clone();
    tl.genome_content_hash = 0;
    let profile = AnimProfile { name: name.to_string(), timeline: tl };
    let json = serde_json::to_string_pretty(&profile)?;
    std::fs::write(profile_path(name), json)?;
    Ok(())
}

/// Loads the named profile, or `None` if it doesn't exist / is unreadable.
pub fn load_profile(name: &str) -> Option<AnimProfile> {
    let json = std::fs::read_to_string(profile_path(name)).ok()?;
    serde_json::from_str(&json).ok()
}

/// Deletes the named profile. Not an error if it was already gone.
pub fn delete_profile(name: &str) -> Result<()> {
    match std::fs::remove_file(profile_path(name)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each test uses its OWN distinctive name — these tests run in
    // parallel by default, and a shared name/file would race (confirmed:
    // an earlier version of this test module that shared one constant name
    // across all four tests failed intermittently for exactly this
    // reason). Matches `anim_persist.rs`'s own convention of a distinct
    // literal per test rather than one shared fixture.

    #[test]
    fn save_then_load_round_trips_and_clears_genome_hash() {
        let name = "__test_profile_round_trip__";
        let _ = std::fs::remove_file(profile_path(name));
        let mut tl = AnimationTimeline::new(0xDEADBEEF);
        tl.delta_t = 2.5;
        tl.duration_s = 12.0;

        save_profile(name, &tl).expect("save should succeed");
        let loaded = load_profile(name).expect("the profile we just saved should load back");
        assert_eq!(loaded.name, name);
        assert_eq!(loaded.timeline.delta_t, 2.5);
        assert_eq!(loaded.timeline.duration_s, 12.0);
        assert_eq!(loaded.timeline.genome_content_hash, 0, "a saved profile must not carry the source genome's hash");

        let _ = std::fs::remove_file(profile_path(name));
    }

    #[test]
    fn save_over_an_existing_profile_replaces_it_not_appends() {
        let name = "__test_profile_save_over__";
        let _ = std::fs::remove_file(profile_path(name));
        let mut tl = AnimationTimeline::new(1);
        tl.delta_t = 1.0;
        save_profile(name, &tl).unwrap();
        tl.delta_t = 3.0;
        save_profile(name, &tl).unwrap();

        let loaded = load_profile(name).unwrap();
        assert_eq!(loaded.timeline.delta_t, 3.0, "second save must overwrite the first");
        assert_eq!(list_profiles().iter().filter(|n| *n == name).count(), 1, "must not create a duplicate entry");

        let _ = std::fs::remove_file(profile_path(name));
    }

    #[test]
    fn delete_removes_it_and_is_not_an_error_if_already_gone() {
        let name = "__test_profile_delete__";
        let _ = std::fs::remove_file(profile_path(name));
        let tl = AnimationTimeline::new(1);
        save_profile(name, &tl).unwrap();
        assert!(load_profile(name).is_some());

        delete_profile(name).expect("delete should succeed");
        assert!(load_profile(name).is_none());
        delete_profile(name).expect("deleting an already-gone profile must not error");
    }

    #[test]
    fn list_profiles_includes_a_freshly_saved_one() {
        let name = "__test_profile_listing__";
        let _ = std::fs::remove_file(profile_path(name));
        let tl = AnimationTimeline::new(1);
        save_profile(name, &tl).unwrap();
        assert!(list_profiles().contains(&name.to_string()));
        let _ = std::fs::remove_file(profile_path(name));
    }

    #[test]
    fn loading_a_name_that_was_never_saved_is_none() {
        assert!(load_profile("__definitely_never_saved_xyz__").is_none());
    }
}
