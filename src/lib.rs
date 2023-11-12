//! `frechet` is a library providing simple dual number types with interfaces
//! similar to that of the standard floating point types. Additionally, provides
//! a small interface to abstract the computation of derivatives.
//!
//! # Example
//! ```
//! use frechet::*;
//!
//! fn p(x: dual32) -> dual32 { x.powf(2.5).atanh() + 1.0  }
//! fn p_derivative(x: f32) -> f32 { -2.5 * x.powf(1.5)/(x.powi(5) - 1.0) }
//!
//! // using the `derivative` function
//! let z1 = derivative(p, 2.0);
//!
//! // manually
//! let z2 = p(2.0.as_dual_variable()).d;
//!
//! // exact derivative
//! let z3 = p_derivative(2.0);
//!
//! assert!((z1 - z3).abs() < f32::EPSILON);
//! assert!((z2 - z3).abs() < f32::EPSILON);
//! ````

/// Type representing a simple dual number, via its "real" and "imaginary" parts,
/// i.e. represents z = x + d*j, where is such that `j != 0.0` and `j*j == 0.0`.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
pub struct dual32 {
    pub x: f32,
    pub d: f32,
}

impl dual32 {
    pub const ZERO: dual32 = dual32 { x: 0.0, d: 0.0 };
    pub const ONE: dual32 = dual32 { x: 1.0, d: 0.0 };

    /// The unit "imaginary" dual number. Satisfies  `J*J == ZERO`.
    pub const J: dual32 = dual32 { x: 0.0, d: 1.0 };

    #[inline]
    pub const fn new(x: f32, d: f32) -> dual32 {
        Self { x, d }
    }
}

impl core::ops::Add<dual32> for dual32 {
    type Output = dual32;

    #[inline]
    fn add(self, rhs: dual32) -> Self::Output {
        Self::Output::new(self.x + rhs.x, self.d + rhs.d)
    }
}

impl core::ops::Add<f32> for dual32 {
    type Output = dual32;

    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        Self::Output::new(self.x + rhs, self.d)
    }
}

impl core::ops::Add<dual32> for f32 {
    type Output = dual32;

    #[inline]
    fn add(self, rhs: dual32) -> Self::Output {
        Self::Output::new(self + rhs.x, rhs.d)
    }
}

impl core::ops::Sub<dual32> for dual32 {
    type Output = dual32;

    #[inline]
    fn sub(self, rhs: dual32) -> Self::Output {
        Self::Output::new(self.x - rhs.x, self.d - rhs.d)
    }
}

impl core::ops::Sub<f32> for dual32 {
    type Output = dual32;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Self::Output::new(self.x - rhs, self.d)
    }
}

impl core::ops::Sub<dual32> for f32 {
    type Output = dual32;

    #[inline]
    fn sub(self, rhs: dual32) -> Self::Output {
        Self::Output::new(self - rhs.x, rhs.d)
    }
}

impl core::ops::Mul<dual32> for dual32 {
    type Output = dual32;

    #[inline]
    fn mul(self, rhs: dual32) -> Self::Output {
        Self::Output::new(self.x * rhs.x, self.x * rhs.d + self.d * rhs.x)
    }
}

impl core::ops::Mul<f32> for dual32 {
    type Output = dual32;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self::Output::new(self.x * rhs, self.x * rhs)
    }
}

impl core::ops::Mul<dual32> for f32 {
    type Output = dual32;

    #[inline]
    fn mul(self, rhs: dual32) -> Self::Output {
        Self::Output::new(self * rhs.x, self * rhs.d)
    }
}

impl core::ops::Div<dual32> for dual32 {
    type Output = dual32;

    #[inline]
    fn div(self, rhs: dual32) -> Self::Output {
        Self::Output::new(
            self.x / rhs.x,
            (self.d * rhs.x - self.x * rhs.d) / (rhs.x * rhs.x),
        )
    }
}

impl core::ops::Div<f32> for dual32 {
    type Output = dual32;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Self::Output::new(self.x / rhs, self.x / rhs)
    }
}

impl core::ops::Div<dual32> for f32 {
    type Output = dual32;

    #[inline]
    fn div(self, rhs: dual32) -> Self::Output {
        Self::Output::new(self / rhs.x, (-self * rhs.d) / (rhs.x * rhs.x))
    }
}

// Power functions
impl dual32 {
    #[inline]
    pub fn recip(self) -> dual32 {
        Self::new(self.x.recip(), -self.d * self.x.recip().powi(2))
    }

    #[inline]
    pub fn powf(self, n: f32) -> dual32 {
        Self::new(self.x.powf(n), self.d * n * self.x.powf(n - 1.0))
    }

    #[inline]
    pub fn powi(self, n: i32) -> dual32 {
        Self::new(self.x.powi(n), self.d * n as f32 * self.x.powi(n - 1))
    }

    #[inline]
    pub fn sqrt(self) -> dual32 {
        Self::new(self.x.sqrt(), self.d * 0.5 / self.x.sqrt())
    }

    #[inline]
    pub fn cbrt(self) -> dual32 {
        Self::new(self.x.cbrt(), self.d * (1.0 / 3.0) / self.x.cbrt().powi(2))
    }
}

// Exponentials and logarithms
impl dual32 {
    #[inline]
    pub fn exp(self) -> dual32 {
        Self::new(self.x.exp(), self.d * self.x.exp())
    }

    #[inline]
    pub fn exp2(self) -> dual32 {
        Self::new(
            self.x.exp2(),
            self.d * core::f32::consts::LN_2 * self.x.exp2(),
        )
    }

    #[inline]
    pub fn exp_m1(self) -> dual32 {
        Self::new(self.x.exp_m1(), self.d * self.x.exp())
    }

    #[inline]
    pub fn ln(self) -> dual32 {
        Self::new(self.x.ln(), self.d / self.x)
    }

    #[inline]
    pub fn ln_1p(self) -> dual32 {
        Self::new(self.x.ln_1p(), self.d / (1.0 + self.x))
    }

    #[inline]
    pub fn log(self, base: f32) -> dual32 {
        Self::new(self.x.log(base), self.d / (base.ln() * self.x))
    }

    #[inline]
    pub fn log10(self) -> dual32 {
        Self::new(self.x.log10(), self.d / (core::f32::consts::LN_10 * self.x))
    }

    #[inline]
    pub fn log2(self) -> dual32 {
        Self::new(self.x.log2(), self.d / (core::f32::consts::LN_2 * self.x))
    }
}

// Trigonometric functions
// atan2 ?
// sin_cos ?
impl dual32 {
    #[inline]
    pub fn cos(self) -> dual32 {
        Self::new(self.x.cos(), -self.d * self.x.sin())
    }

    #[inline]
    pub fn sin(self) -> dual32 {
        Self::new(self.x.sin(), self.d * self.x.cos())
    }

    #[inline]
    pub fn tan(self) -> dual32 {
        Self::new(self.x.tan(), self.d / self.x.cos().powi(2))
    }

    #[inline]
    pub fn acos(self) -> dual32 {
        Self::new(self.x.acos(), -self.d / (1.0 - self.x * self.x).sqrt())
    }

    #[inline]
    pub fn asin(self) -> dual32 {
        Self::new(self.x.asin(), self.d / (1.0 - self.x * self.x).sqrt())
    }

    #[inline]
    pub fn atan(self) -> dual32 {
        Self::new(self.x.tan(), self.d / (1.0 + self.x * self.x))
    }
}

// Hyperbolic functions
impl dual32 {
    #[inline]
    pub fn cosh(self) -> dual32 {
        Self::new(self.x.cosh(), self.d * self.x.sinh())
    }

    #[inline]
    pub fn sinh(self) -> dual32 {
        Self::new(self.x.sinh(), self.d * self.x.cosh())
    }

    #[inline]
    pub fn tanh(self) -> dual32 {
        Self::new(self.x.tanh(), self.d / self.x.cosh().powi(2))
    }

    #[inline]
    pub fn acosh(self) -> dual32 {
        Self::new(self.x.acosh(), self.d / (self.x * self.x - 1.0).sqrt())
    }

    #[inline]
    pub fn asinh(self) -> dual32 {
        Self::new(self.x.asinh(), self.d / (self.x * self.x + 1.0).sqrt())
    }

    #[inline]
    pub fn atanh(self) -> dual32 {
        Self::new(self.x.atanh(), self.d / (1.0 - self.x * self.x))
    }
}

// Continuous differentiable almost-everywhere functions
impl dual32 {
    #[inline]
    pub fn abs(self) -> dual32 {
        Self::new(self.x.abs(), self.d * self.x.signum())
    }

    #[inline]
    pub fn min(self, other: f32) -> dual32 {
        Self::new(
            self.x.min(other),
            if self.x < other { self.d * 1.0 } else { 0.0 },
        )
    }

    #[inline]
    pub fn max(self, other: f32) -> dual32 {
        Self::new(
            self.x.max(other),
            if other < self.x { self.d * 1.0 } else { 0.0 },
        )
    }

    #[inline]
    pub fn clamp(self, min: f32, max: f32) -> dual32 {
        Self::new(
            self.x.clamp(min, max),
            if min < self.x && self.x < max {
                self.d * 1.0
            } else {
                0.0
            },
        )
    }
}

/// Piece-wise constant functions
impl dual32 {
    #[inline]
    pub fn ceil(self) -> dual32 {
        Self::new(self.x.ceil(), 0.0)
    }

    #[inline]
    pub fn floor(self) -> dual32 {
        Self::new(self.x.floor(), 0.0)
    }

    // not constant but x.fract() == x - x.floor()
    #[inline]
    pub fn fract(self) -> dual32 {
        Self::new(self.x.fract(), self.d)
    }

    #[inline]
    pub fn round(self) -> dual32 {
        Self::new(self.x.round(), 0.0)
    }

    #[inline]
    pub fn signum(self) -> dual32 {
        Self::new(self.x.signum(), 0.0)
    }

    #[inline]
    pub fn trunc(self) -> dual32 {
        Self::new(self.x.trunc(), 0.0)
    }
}

/// Sporadic functions
impl dual32 {
    #[inline]
    pub fn mul_add(self, a: f32, b: f32) -> dual32 {
        Self::new(self.x.mul_add(a, b), self.d * a * self.x)
    }
}

/// Trait used to convert a non-dual numeric type into a dual "variable",
/// i.e. a dual number with unit imaginary part.
pub trait AsDualVariable: Copy {
    fn as_dual_variable(self) -> dual32;
    fn from_dual(z: dual32) -> Self;
}

impl AsDualVariable for f32 {
    fn as_dual_variable(self) -> dual32 {
        dual32::new(self, 1.0)
    }

    fn from_dual(z: dual32) -> Self {
        z.x
    }
}

impl AsDualVariable for dual32 {
    fn as_dual_variable(self) -> dual32 {
        self
    }

    fn from_dual(z: dual32) -> Self {
        z
    }
}

/// Computes the derivative of a function `f` at a point `x`
pub fn derivative(f: impl Fn(dual32) -> dual32, x: f32) -> f32 {
    let x = x.as_dual_variable();
    let dual32 { x: _fx, d: fpx } = f(x);
    return fpx;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        let dual32 { x, d } = dual32::J.powi(2);
        assert!(x.abs() < f32::EPSILON);
        assert!(d.abs() < f32::EPSILON);
    }

    #[test]
    fn polynomial() {
        fn p(x: dual32) -> dual32 {
            4.0 * x * x - 3.0 * x + 3.0
        }
        let x = 3.0;
        let dual32 { x: px, d: pdx } = p(x + dual32::J);

        assert!((px - 30.0).abs() <= f32::EPSILON);
        assert!((pdx - 21.0).abs() <= f32::EPSILON);
    }
}
