import Mathlib

/--
For all real `a b`, we have `(a + b)^2 ≤ 2*a^2 + 2*b^2`.

Proof idea:
- Start from `0 ≤ (a - b)^2`.
- Expand to get `0 ≤ a^2 - 2ab + b^2`, then rearrange to `2ab ≤ a^2 + b^2`.
- Add `a^2 + b^2` to both sides and rewrite the left as `(a + b)^2`.
- Use `ring` to normalize polynomial expressions.
-/
theorem sq_add_le_two_sq (a b : ℝ) : (a + b)^2 ≤ 2*a^2 + 2*b^2 := by
  -- Step 1: nonnegativity of a square
  have h0 : 0 ≤ (a - b) ^ 2 := by
    exact sq_nonneg (a - b)
  -- Expand (a - b)^2 to a^2 - 2ab + b^2
  have h1 : 0 ≤ a^2 - 2*a*b + b^2 := by
    have hs : (a - b) ^ 2 = a^2 - 2*a*b + b^2 := by ring
    simpa [hs] using h0
  -- From h1, add 2ab to both sides to get 2ab ≤ a^2 + b^2
  have twoab_le : 2*a*b ≤ a^2 + b^2 := by
    -- Add 2ab to both sides of h1 and simplify
    have := add_le_add_left h1 (2*a*b)
    -- LHS: 2ab + 0, RHS: 2ab + (a^2 - 2ab + b^2)
    simpa [add_comm, add_left_comm, add_assoc] using this
  -- Step 2: turn the goal into that inequality plus some algebra
  calc
    (a + b)^2 = a^2 + 2*a*b + b^2 := by ring
    _ ≤ a^2 + (a^2 + b^2) + b^2 := by
      -- Add a^2 and b^2 to both sides of 2ab ≤ a^2 + b^2
      have := add_le_add_left (add_le_add_right twoab_le (b^2 : ℝ)) (a^2 : ℝ)
      simpa [add_assoc] using this
    _ = 2*a^2 + 2*b^2 := by ring
