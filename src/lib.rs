//! `frechet` is a library providing simple dual number types with interfaces
//! similar to that of the standard floating point types. Additionally, provides
//! a small interface to abstract the computation of derivatives.
//!
//! # Refresher on dual numbers
//!
//! A dual number $z$ is represented as the sum of a "real part" $x$ and an "imaginary part" $y$,
//! and written $z = x + yj$, where $j$ is purely symbolic and follows the rule $j^2 = 0$.
//! As an example, evaluating the polynomial $P(X)=3X^2-X+1$ at the dual number $X + j$ yields
//! $$P(X+j)=3X^2-X+1 + (6X-1)j = P(X) + P^\prime(X)j.$$
//! This motivates the following extension of any (differentiable) real function $f$ to dual numbers:
//! $$f(x+yj) = f(x) + yf^\prime(x)j.$$
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
//! // doing it manually (strictly equivalent)
//! let z2 = p(2.0f32.as_dual_variable()).im;
//!
//! // exact derivative
//! let z3 = p_derivative(2.0);
//!
//! assert!((z1 - z3).abs() < f32::EPSILON);
//! assert!((z2 - z3).abs() < f32::EPSILON);
//! ```

mod dual;
pub use dual::{dual32, dual64, Dual};

/// Trait allowing conversion of non-dual types into the corresponding "dual variable",
/// i.e. with unit imaginy part.
pub trait AsDualVariable: Sized {
    type Precision;
    fn as_dual_variable(self) -> Dual<Self::Precision>;
}

impl AsDualVariable for f32 {
    type Precision = f32;

    fn as_dual_variable(self) -> dual32 {
        dual32::new(self, 1.0)
    }
}

impl AsDualVariable for f64 {
    type Precision = f64;

    fn as_dual_variable(self) -> dual64 {
        dual64::new(self, 1.0)
    }
}

impl<F> AsDualVariable for Dual<F> {
    type Precision = F;

    fn as_dual_variable(self) -> Self {
        self
    }
}

/// Computes the derivative of a function `f` at a point `x`
pub fn derivative<F>(f: impl Fn(Dual<F>) -> Dual<F>, x: F) -> F
where
    F: AsDualVariable<Precision = F>,
{
    f(x.as_dual_variable()).im
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity32() {
        let dual32 { re, im } = dual32::J.powi(2);
        assert!(re.abs() < f32::EPSILON);
        assert!(im.abs() < f32::EPSILON);
    }

    #[test]
    fn identity64() {
        let dual64 { re, im } = dual64::J.powi(2);
        assert!(re.abs() < f64::EPSILON);
        assert!(im.abs() < f64::EPSILON);
    }

    #[test]
    fn polynomial32() {
        fn p(x: dual32) -> dual32 {
            4.0 * x * x - 3.0 * x + 3.0
        }
        let x = 3.0;
        let dual32 { re, im } = p(x + dual32::J);

        assert!((re - 30.0).abs() <= f32::EPSILON);
        assert!((im - 21.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn polynomial64() {
        fn p(x: dual64) -> dual64 {
            4.0 * x * x - 3.0 * x + 3.0
        }
        let x = 3.0;
        let dual64 { re, im } = p(x + dual64::J);

        assert!((re - 30.0).abs() <= f64::EPSILON);
        assert!((im - 21.0).abs() <= f64::EPSILON);
    }
}
