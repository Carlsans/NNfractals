//! 3D-rotation quaternion — used ONLY for composing camera orientation
//! from Rotation-effect tracks (animation-viewer plan, Phase 5).
//!
//! Naming: `Quat`/`quaternion` already means exactly one thing elsewhere
//! in this codebase — the fractal's own 4D iteration parameter
//! (`crate::quaternion::Quat{r,a,b,c}`, mirrored in every `.wgsl` shader).
//! This type is deliberately named `OrientQuat`, not `Quat`, and lives in
//! its own module, so "quaternion as the 4D fractal domain" and
//! "quaternion as a 3D rotation" are never confused in this codebase.
//! Nothing here is aware of `crate::quaternion::Quat` and nothing there is
//! aware of this — they're unrelated types that happen to share the same
//! underlying algebra, exactly like `f64` and `Vec3` do.

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OrientQuat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl OrientQuat {
    pub const IDENTITY: OrientQuat = OrientQuat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    /// `axis` need not be unit length (normalized internally); the zero
    /// vector maps to the identity rotation rather than panicking or
    /// producing NaN, since `accumulated_angle` can legitimately be 0.0
    /// for an empty track and axis choice is then irrelevant.
    pub fn from_axis_angle(axis: (f64, f64, f64), angle_rad: f64) -> Self {
        let len = (axis.0 * axis.0 + axis.1 * axis.1 + axis.2 * axis.2).sqrt();
        let (ax, ay, az) = if len > 1e-12 { (axis.0 / len, axis.1 / len, axis.2 / len) } else { (0.0, 0.0, 0.0) };
        let half = angle_rad * 0.5;
        let s = half.sin();
        OrientQuat { w: half.cos(), x: ax * s, y: ay * s, z: az * s }
    }

    /// Hamilton product `self * other` — composing `a.mul(&b)` means
    /// "apply `b` first, then `a`" (standard quaternion-rotation
    /// convention), which is why `camera_orientation`'s X->Y->Z documented
    /// order is written `qz.mul(&qy).mul(&qx)`.
    pub fn mul(&self, other: &OrientQuat) -> OrientQuat {
        OrientQuat {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Rotates the vector `v` by this orientation: `q * (0,v) * q_conj`.
    pub fn rotate(&self, v: (f64, f64, f64)) -> (f64, f64, f64) {
        let qv = OrientQuat { w: 0.0, x: v.0, y: v.1, z: v.2 };
        let conj = OrientQuat { w: self.w, x: -self.x, y: -self.y, z: -self.z };
        let r = self.mul(&qv).mul(&conj);
        (r.x, r.y, r.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: (f64, f64, f64), b: (f64, f64, f64), eps: f64) -> bool {
        (a.0 - b.0).abs() < eps && (a.1 - b.1).abs() < eps && (a.2 - b.2).abs() < eps
    }

    #[test]
    fn identity_rotation_is_a_no_op() {
        let v = (1.0, 2.0, -3.0);
        assert!(approx(OrientQuat::IDENTITY.rotate(v), v, 1e-12));
    }

    #[test]
    fn zero_angle_is_the_identity_regardless_of_axis() {
        let q = OrientQuat::from_axis_angle((1.0, 1.0, 1.0), 0.0);
        let v = (0.3, -0.7, 2.0);
        assert!(approx(q.rotate(v), v, 1e-9));
    }

    #[test]
    fn ninety_degrees_around_z_maps_x_to_y() {
        let q = OrientQuat::from_axis_angle((0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
        let r = q.rotate((1.0, 0.0, 0.0));
        assert!(approx(r, (0.0, 1.0, 0.0), 1e-9), "got {r:?}");
    }

    #[test]
    fn ninety_degrees_around_x_maps_y_to_z() {
        let q = OrientQuat::from_axis_angle((1.0, 0.0, 0.0), std::f64::consts::FRAC_PI_2);
        let r = q.rotate((0.0, 1.0, 0.0));
        assert!(approx(r, (0.0, 0.0, 1.0), 1e-9), "got {r:?}");
    }

    #[test]
    fn composing_two_quarter_turns_matches_one_half_turn() {
        let quarter = OrientQuat::from_axis_angle((0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
        let half = OrientQuat::from_axis_angle((0.0, 0.0, 1.0), std::f64::consts::PI);
        let composed = quarter.mul(&quarter);
        let v = (1.0, 0.0, 0.0);
        assert!(approx(composed.rotate(v), half.rotate(v), 1e-9));
    }

    #[test]
    fn a_non_unit_axis_is_normalized_internally() {
        let q1 = OrientQuat::from_axis_angle((0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
        let q2 = OrientQuat::from_axis_angle((0.0, 0.0, 5.0), std::f64::consts::FRAC_PI_2);
        let v = (1.0, 0.0, 0.0);
        assert!(approx(q1.rotate(v), q2.rotate(v), 1e-9));
    }

    #[test]
    fn rotation_preserves_vector_length() {
        let q = OrientQuat::from_axis_angle((0.3, 0.6, 0.1), 1.234);
        let v = (2.0, -1.5, 0.7);
        let r = q.rotate(v);
        let len_before = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
        let len_after = (r.0 * r.0 + r.1 * r.1 + r.2 * r.2).sqrt();
        assert!((len_before - len_after).abs() < 1e-9, "before={len_before} after={len_after}");
    }
}
