use core::ops::{Add, Div, Mul, Sub};
use num_traits::{Float, FloatConst, FromPrimitive, One, Zero};

/// Type representing a dual number, which precision is determined by
/// the type variable `F`.
#[derive(Clone, Copy, Debug)]
pub struct Dual<F> {
    pub re: F,
    pub im: F,
}

/// Type representing a 32-bit precision dual number
#[allow(non_camel_case_types)]
pub type dual32 = Dual<f32>;

impl dual32 {
    pub const ZERO: Self = Self::new(0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 0.0);
    /// The purely imaginary unit dual number. Satisfies `J*J == ZERO`
    pub const J: Self = Self::new(0.0, 1.0);
}

/// Type representing a 64-bit precision dual number
#[allow(non_camel_case_types)]
pub type dual64 = Dual<f64>;

impl dual64 {
    pub const ZERO: Self = Self::new(0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 0.0);
    /// The purely imaginary unit dual number. Satisfies `J*J == ZERO`
    pub const J: Self = Self::new(0.0, 1.0);
}

impl<F> Dual<F> {
    #[inline]
    pub const fn new(re: F, im: F) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn zero() -> Self
    where
        F: Zero,
    {
        Self::new(F::zero(), F::zero())
    }

    #[inline]
    pub fn one() -> Self
    where
        F: Zero + One,
    {
        Self::new(F::one(), F::zero())
    }

    #[inline]
    pub fn j() -> Self
    where
        F: Zero + One,
    {
        Self::new(F::zero(), F::one())
    }
}

impl<F> Add<Dual<F>> for Dual<F>
where
    F: Copy + Add<F, Output = F>,
{
    type Output = Dual<F>;

    #[inline]
    fn add(self, rhs: Dual<F>) -> Self::Output {
        Self::Output::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<F> Add<F> for Dual<F>
where
    F: Copy + Add<F, Output = F>,
{
    type Output = Dual<F>;

    #[inline]
    fn add(self, rhs: F) -> Self::Output {
        Self::Output::new(self.re + rhs, self.im)
    }
}

impl Add<dual32> for f32 {
    type Output = dual32;

    #[inline]
    fn add(self, rhs: dual32) -> Self::Output {
        Self::Output::new(self + rhs.re, rhs.im)
    }
}

impl Add<dual64> for f64 {
    type Output = dual64;

    #[inline]
    fn add(self, rhs: dual64) -> Self::Output {
        Self::Output::new(self + rhs.re, rhs.im)
    }
}

impl<F> Sub<Dual<F>> for Dual<F>
where
    F: Copy + Sub<F, Output = F>,
{
    type Output = Dual<F>;

    #[inline]
    fn sub(self, rhs: Dual<F>) -> Self::Output {
        Self::Output::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<F> Sub<F> for Dual<F>
where
    F: Copy + Sub<F, Output = F>,
{
    type Output = Dual<F>;

    #[inline]
    fn sub(self, rhs: F) -> Self::Output {
        Self::Output::new(self.re - rhs, self.im)
    }
}

impl Sub<dual32> for f32 {
    type Output = dual32;

    #[inline]
    fn sub(self, rhs: dual32) -> Self::Output {
        Self::Output::new(self - rhs.re, rhs.im)
    }
}

impl Sub<dual64> for f64 {
    type Output = dual64;

    #[inline]
    fn sub(self, rhs: dual64) -> Self::Output {
        Self::Output::new(self - rhs.re, rhs.im)
    }
}

impl<F> Mul<Dual<F>> for Dual<F>
where
    F: Copy + Mul<F, Output = F> + Add<F, Output = F>,
{
    type Output = Dual<F>;

    #[inline]
    fn mul(self, rhs: Dual<F>) -> Self::Output {
        Self::Output::new(self.re * rhs.re, self.re * rhs.im + self.im * rhs.re)
    }
}

impl<F> Mul<F> for Dual<F>
where
    F: Copy + Mul<F, Output = F>,
{
    type Output = Dual<F>;

    #[inline]
    fn mul(self, rhs: F) -> Self::Output {
        Self::Output::new(self.re * rhs, self.re * rhs)
    }
}

impl Mul<dual32> for f32 {
    type Output = dual32;

    #[inline]
    fn mul(self, rhs: dual32) -> Self::Output {
        Self::Output::new(self * rhs.re, self * rhs.im)
    }
}

impl Mul<dual64> for f64 {
    type Output = dual64;

    #[inline]
    fn mul(self, rhs: dual64) -> Self::Output {
        Self::Output::new(self * rhs.re, self * rhs.im)
    }
}

impl<F> Div<Dual<F>> for Dual<F>
where
    F: Copy + Div<F, Output = F> + Mul<F, Output = F> + Sub<F, Output = F>,
{
    type Output = Dual<F>;

    #[inline]
    fn div(self, rhs: Dual<F>) -> Self::Output {
        Self::Output::new(
            self.re / rhs.re,
            (self.im * rhs.re - self.re * rhs.im) / (rhs.re * rhs.re),
        )
    }
}

impl<F> Div<F> for Dual<F>
where
    F: Copy + Div<F, Output = F>,
{
    type Output = Dual<F>;

    #[inline]
    fn div(self, rhs: F) -> Self::Output {
        Self::Output::new(self.re / rhs, self.re / rhs)
    }
}

impl Div<dual32> for f32 {
    type Output = dual32;

    #[inline]
    fn div(self, rhs: dual32) -> Self::Output {
        Self::Output::new(self / rhs.re, (-self * rhs.im) / (rhs.re * rhs.re))
    }
}

impl Div<dual64> for f64 {
    type Output = dual64;

    #[inline]
    fn div(self, rhs: dual64) -> Self::Output {
        Self::Output::new(self / rhs.re, (-self * rhs.im) / (rhs.re * rhs.re))
    }
}

/// "Power" functions
impl<F: Float + FromPrimitive> Dual<F> {
    #[inline]
    pub fn recip(self) -> Dual<F> {
        Self::new(self.re.recip(), -self.im * self.re.recip().powi(2))
    }

    #[inline]
    pub fn powf(self, n: F) -> Dual<F> {
        Self::new(self.re.powf(n), self.im * n * self.re.powf(n - F::one()))
    }

    #[inline]
    pub fn powi(self, n: i32) -> Dual<F> {
        Self::new(
            self.re.powi(n),
            self.im * F::from_i32(n).unwrap() * self.re.powi(n - 1),
        )
    }

    #[inline]
    pub fn sqrt(self) -> Dual<F> {
        Self::new(
            self.re.sqrt(),
            self.im * F::from_f64(0.5).unwrap() / self.re.sqrt(),
        )
    }

    #[inline]
    pub fn cbrt(self) -> Dual<F> {
        Self::new(
            self.re.cbrt(),
            self.im * F::from_f64(1.0 / 3.0).unwrap() / self.re.cbrt().powi(2),
        )
    }
}

/// Exponential functions
impl<F: Float + FloatConst> Dual<F> {
    #[inline]
    pub fn exp(self) -> Dual<F> {
        Self::new(self.re.exp(), self.im * self.re.exp())
    }

    #[inline]
    pub fn exp2(self) -> Dual<F> {
        Self::new(self.re.exp2(), self.im * F::LN_2() * self.re.exp2())
    }

    #[inline]
    pub fn exp_m1(self) -> Dual<F> {
        Self::new(self.re.exp_m1(), self.im * self.re.exp())
    }

    #[inline]
    pub fn ln(self) -> Dual<F> {
        Self::new(self.re.ln(), self.im / self.re)
    }

    #[inline]
    pub fn ln_1p(self) -> Dual<F> {
        Self::new(self.re.ln_1p(), self.im / (F::one() + self.re))
    }

    #[inline]
    pub fn log(self, base: F) -> Dual<F> {
        Self::new(self.re.log(base), self.im / (base.ln() * self.re))
    }

    #[inline]
    pub fn log10(self) -> Dual<F> {
        Self::new(self.re.log10(), self.im / (F::LN_10() * self.re))
    }

    #[inline]
    pub fn log2(self) -> Dual<F> {
        Self::new(self.re.log2(), self.im / (F::LN_2() * self.re))
    }
}

/// Trigonometric functions
impl<F: Float> Dual<F> {
    #[inline]
    pub fn cos(self) -> Dual<F> {
        Self::new(self.re.cos(), -self.im * self.re.sin())
    }

    #[inline]
    pub fn sin(self) -> Dual<F> {
        Self::new(self.re.sin(), self.im * self.re.cos())
    }

    #[inline]
    pub fn tan(self) -> Dual<F> {
        Self::new(self.re.tan(), self.im / self.re.cos().powi(2))
    }

    #[inline]
    pub fn acos(self) -> Dual<F> {
        Self::new(
            self.re.acos(),
            -self.im / (F::one() - self.re * self.re).sqrt(),
        )
    }

    #[inline]
    pub fn asin(self) -> Dual<F> {
        Self::new(
            self.re.asin(),
            self.im / (F::one() - self.re * self.re).sqrt(),
        )
    }

    #[inline]
    pub fn atan(self) -> Dual<F> {
        Self::new(self.re.tan(), self.im / (F::one() + self.re * self.re))
    }
}

/// Hyperbolic functions
impl<F: Float> Dual<F> {
    #[inline]
    pub fn cosh(self) -> Dual<F> {
        Self::new(self.re.cosh(), self.im * self.re.sinh())
    }

    #[inline]
    pub fn sinh(self) -> Dual<F> {
        Self::new(self.re.sinh(), self.im * self.re.cosh())
    }

    #[inline]
    pub fn tanh(self) -> Dual<F> {
        Self::new(self.re.tanh(), self.im / self.re.cosh().powi(2))
    }

    #[inline]
    pub fn acosh(self) -> Dual<F> {
        Self::new(
            self.re.acosh(),
            self.im / (self.re * self.re - F::one()).sqrt(),
        )
    }

    #[inline]
    pub fn asinh(self) -> Dual<F> {
        Self::new(
            self.re.asinh(),
            self.im / (self.re * self.re + F::one()).sqrt(),
        )
    }

    #[inline]
    pub fn atanh(self) -> Dual<F> {
        Self::new(self.re.atanh(), self.im / (F::one() - self.re * self.re))
    }
}

/// Lipschitz continuous functions
impl<F: Float> Dual<F> {
    #[inline]
    pub fn abs(self) -> Dual<F> {
        Self::new(self.re.abs(), self.im * self.re.signum())
    }

    #[inline]
    pub fn min(self, other: F) -> Dual<F> {
        Self::new(
            self.re.min(other),
            if self.re < other { self.im } else { F::zero() },
        )
    }

    #[inline]
    pub fn max(self, other: F) -> Dual<F> {
        Self::new(
            self.re.max(other),
            if other < self.re { self.im } else { F::zero() },
        )
    }

    // #[inline]
    // pub fn clamp(self, min: F, max: F) -> Dual<F> {
    //     Self::new(
    //         self.re.clamp(min, max),
    //         if min < self.re && self.re < max {
    //             self.im
    //         } else {
    //             F::zero()
    //         },
    //     )
    // }
}

/// Piecewise constant functions
impl<F: Float> Dual<F> {
    #[inline]
    pub fn ceil(self) -> Dual<F> {
        Self::new(self.re.ceil(), F::zero())
    }

    #[inline]
    pub fn floor(self) -> Dual<F> {
        Self::new(self.re.floor(), F::zero())
    }

    // not constant but x.fract() == x - x.floor()
    #[inline]
    pub fn fract(self) -> Dual<F> {
        Self::new(self.re.fract(), self.im)
    }

    #[inline]
    pub fn round(self) -> Dual<F> {
        Self::new(self.re.round(), F::zero())
    }

    #[inline]
    pub fn signum(self) -> Dual<F> {
        Self::new(self.re.signum(), F::zero())
    }

    #[inline]
    pub fn trunc(self) -> Dual<F> {
        Self::new(self.re.trunc(), F::zero())
    }
}
