//! Minimal quaternion arithmetic for the quaternion-Mandelbrot prototype
//! (see `quat_fractal`/`quat_motion`). Carl's own component naming is used:
//! a quaternion is R + A·i + B·j + C·k.

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Quat {
    pub r: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Quat {
    pub const ZERO: Quat = Quat { r: 0.0, a: 0.0, b: 0.0, c: 0.0 };

    pub const fn new(r: f64, a: f64, b: f64, c: f64) -> Self {
        Quat { r, a, b, c }
    }

    #[inline]
    pub fn add(self, o: Quat) -> Quat {
        Quat::new(self.r + o.r, self.a + o.a, self.b + o.b, self.c + o.c)
    }

    #[inline]
    pub fn sub(self, o: Quat) -> Quat {
        Quat::new(self.r - o.r, self.a - o.a, self.b - o.b, self.c - o.c)
    }

    /// Hamilton product: i²=j²=k²=-1, ij=k, jk=i, ki=j (anticommutative).
    #[inline]
    pub fn mul(self, o: Quat) -> Quat {
        Quat {
            r: self.r * o.r - self.a * o.a - self.b * o.b - self.c * o.c,
            a: self.r * o.a + self.a * o.r + self.b * o.c - self.c * o.b,
            b: self.r * o.b - self.a * o.c + self.b * o.r + self.c * o.a,
            c: self.r * o.c + self.a * o.b - self.b * o.a + self.c * o.r,
        }
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.r * self.r + self.a * self.a + self.b * self.b + self.c * self.c
    }

    /// Conjugate: negates the vector part (a,b,c), keeps r — the quaternion
    /// generalization of complex conjugation, used by the Tricorn family.
    #[inline]
    pub fn conj(self) -> Quat {
        Quat::new(self.r, -self.a, -self.b, -self.c)
    }

    /// Absolute value of every component — the quaternion generalization of
    /// the Burning Ship fold `|Re z| + i|Im z|`.
    #[inline]
    pub fn abs_components(self) -> Quat {
        Quat::new(self.r.abs(), self.a.abs(), self.b.abs(), self.c.abs())
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.r.is_finite() && self.a.is_finite() && self.b.is_finite() && self.c.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const I: Quat = Quat::new(0.0, 1.0, 0.0, 0.0);
    const J: Quat = Quat::new(0.0, 0.0, 1.0, 0.0);
    const K: Quat = Quat::new(0.0, 0.0, 0.0, 1.0);
    const NEG_ONE: Quat = Quat::new(-1.0, 0.0, 0.0, 0.0);

    #[test]
    fn i_squared_is_minus_one() {
        assert_eq!(I.mul(I), NEG_ONE);
    }

    #[test]
    fn j_squared_is_minus_one() {
        assert_eq!(J.mul(J), NEG_ONE);
    }

    #[test]
    fn k_squared_is_minus_one() {
        assert_eq!(K.mul(K), NEG_ONE);
    }

    #[test]
    fn i_times_j_is_k() {
        assert_eq!(I.mul(J), K);
    }

    #[test]
    fn j_times_k_is_i() {
        assert_eq!(J.mul(K), I);
    }

    #[test]
    fn k_times_i_is_j() {
        assert_eq!(K.mul(I), J);
    }

    #[test]
    fn j_times_i_is_minus_k() {
        assert_eq!(J.mul(I), Quat::new(0.0, 0.0, 0.0, -1.0));
    }

    #[test]
    fn restricted_to_r_a_plane_matches_complex_multiplication() {
        let p = Quat::new(1.3, -0.7, 0.0, 0.0);
        let got = p.mul(p);
        let want = Quat::new(1.3 * 1.3 - (-0.7) * (-0.7), 2.0 * 1.3 * -0.7, 0.0, 0.0);
        assert_eq!(got, want);
        assert_eq!(got.b, 0.0);
        assert_eq!(got.c, 0.0);
    }

    #[test]
    fn norm_sq_matches_expected() {
        let q = Quat::new(1.0, 2.0, 2.0, 4.0);
        assert_eq!(q.norm_sq(), 1.0 + 4.0 + 4.0 + 16.0);
    }

    #[test]
    fn zero_is_additive_identity() {
        let q = Quat::new(0.4, -1.2, 3.3, 0.0);
        assert_eq!(q.add(Quat::ZERO), q);
    }

    #[test]
    fn conj_negates_vector_part_only() {
        let q = Quat::new(1.5, 2.0, -3.0, 4.0);
        assert_eq!(q.conj(), Quat::new(1.5, -2.0, 3.0, -4.0));
    }

    #[test]
    fn abs_components_is_componentwise_abs() {
        let q = Quat::new(-1.5, 2.0, -3.0, -4.0);
        assert_eq!(q.abs_components(), Quat::new(1.5, 2.0, 3.0, 4.0));
    }
}
