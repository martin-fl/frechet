# frechet

`frechet` implements the simple dual number types `dual32` and `dual64`, thus 
providing basic auto-differentiation capabilities.

# Example 

```rust
use frechet::*;

fn p(x: dual32) -> dual32 { x.powf(2.5).atanh() + 1.0  }
fn p_derivative(x: f32) -> f32 { -2.5 * x.powf(1.5)/(x.powi(5) - 1.0) }

// using the `derivative` function
let z1 = derivative(p, 2.0);

// manually
let z2 = p(2.0.as_dual_variable()).d;

// exact derivative
let z3 = p_derivative(2.0);

assert!((z1 - z3).abs() < f32::EPSILON);
assert!((z2 - z3).abs() < f32::EPSILON);
```
