module

public import BoundsExtendedCore

@[expose] public section

noncomputable section

namespace PearsonClassification

def confidenceMultiplier (α : ℝ) : ℝ := 1 + Real.sqrt (Real.log (1 / α))

lemma confidenceMultiplier_pos (α : ℝ) : 0 < confidenceMultiplier α := by
  unfold confidenceMultiplier
  positivity

theorem nullBound_explicit {K reg α : ℝ} {n : ℕ} (hK : 0 ≤ K) :
    nullBound K n reg α = K / ((n : ℝ) * reg) * confidenceMultiplier α ^ 2 := by
  unfold nullBound upperBound meanRadius confidenceMultiplier
  rw [mul_pow, Real.sq_sqrt (by positivity)]
  ring

def separationRequirement (K : ℝ) (n : ℕ) (reg α₀ α₁ : ℝ) : ℝ :=
  K / ((n : ℝ) * reg) *
    (confidenceMultiplier (α₁ / 2) + confidenceMultiplier α₀ *
      Real.sqrt (1 + K / (reg * Real.sqrt (n : ℝ)) * confidenceMultiplier (α₁ / 2))) ^ 2

theorem separationRequirement_eq {K reg α₀ α₁ : ℝ} {n : ℕ} (hK : 0 ≤ K) :
    separationRequirement K n reg α₀ α₁ =
      (meanRadius K n (α₁ / 2) +
        Real.sqrt (1 + operatorRadius K n (α₁ / 2) / reg) * meanRadius K n α₀) ^ 2 /
          (2 * reg) := by
  have hfactor : 1 + operatorRadius K n (α₁ / 2) / reg =
      1 + K / (reg * Real.sqrt (n : ℝ)) * confidenceMultiplier (α₁ / 2) := by
    unfold operatorRadius confidenceMultiplier
    ring
  rw [hfactor]
  unfold meanRadius
  change separationRequirement K n reg α₀ α₁ =
    (Real.sqrt (2 * K / (n : ℝ)) * confidenceMultiplier (α₁ / 2) +
      Real.sqrt (1 + K / (reg * Real.sqrt (n : ℝ)) * confidenceMultiplier (α₁ / 2)) *
        (Real.sqrt (2 * K / (n : ℝ)) * confidenceMultiplier α₀)) ^ 2 / (2 * reg)
  have hsum (r s a b : ℝ) : r * a + s * (r * b) = r * (a + b * s) := by ring
  rw [hsum, mul_pow, Real.sq_sqrt (by positivity)]
  unfold separationRequirement
  ring

theorem explicit_calibration_iff {δ K reg α₀ α₁ : ℝ} {n : ℕ}
    (hδ : 0 ≤ δ) (hK : 0 < K) (hn : 0 < n) (hreg : 0 < reg) :
    (∃ η : ℝ, nullBound K n reg α₀ ≤ η ∧ η < alternativeBound δ K n reg α₁) ↔
      separationRequirement K n reg α₀ α₁ < δ := by
  rw [calibrated_threshold_iff hδ hK hn hreg, separationRequirement_eq hK.le]

/-- The common-tolerance expansion uses `log (2 / α)` exactly as in the TeX. -/
theorem confidenceMultiplier_half (α : ℝ) :
    confidenceMultiplier (α / 2) = 1 + Real.sqrt (Real.log (2 / α)) := by
  unfold confidenceMultiplier
  congr 3
  ring

theorem sufficient_batch_size {δ K reg α : ℝ} {n : ℕ}
    (hδ : 0 < δ) (hK : 0 < K) (hreg : 0 < reg) (hn : 0 < n)
    (hfirst : (K * confidenceMultiplier (α / 2) / reg) ^ 2 ≤ (n : ℝ))
    (hsecond : K * (confidenceMultiplier (α / 2) +
      Real.sqrt 2 * confidenceMultiplier α) ^ 2 / (reg * δ) < (n : ℝ)) :
    ∃ η : ℝ, nullBound K n reg α ≤ η ∧ η < alternativeBound δ K n reg α := by
  apply (explicit_calibration_iff hδ.le hK hn hreg).2
  have hnR : 0 < (n : ℝ) := by exact_mod_cast hn
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.2 hnR
  have ha₀ := confidenceMultiplier_pos α
  have ha₁ := confidenceMultiplier_pos (α / 2)
  have hroot : K * confidenceMultiplier (α / 2) / reg ≤ Real.sqrt (n : ℝ) :=
    (Real.le_sqrt (by positivity) hnR.le).2 hfirst
  have hprod := (div_le_iff₀ hreg).1 hroot
  have he : K / (reg * Real.sqrt (n : ℝ)) * confidenceMultiplier (α / 2) ≤ 1 := by
    rw [div_mul_eq_mul_div]
    apply (div_le_iff₀ (mul_pos hreg hs)).2
    nlinarith [hprod]
  have hc : Real.sqrt (1 + K / (reg * Real.sqrt (n : ℝ)) * confidenceMultiplier (α / 2)) ≤
      Real.sqrt 2 := Real.sqrt_le_sqrt (by linarith)
  have hm := mul_le_mul_of_nonneg_left hc ha₀.le
  have hsum : confidenceMultiplier (α / 2) + confidenceMultiplier α *
      Real.sqrt (1 + K / (reg * Real.sqrt (n : ℝ)) * confidenceMultiplier (α / 2)) ≤
      confidenceMultiplier (α / 2) + Real.sqrt 2 * confidenceMultiplier α := by linarith
  have hsq := square_mono (by positivity : 0 ≤ confidenceMultiplier (α / 2) +
    confidenceMultiplier α * Real.sqrt
      (1 + K / (reg * Real.sqrt (n : ℝ)) * confidenceMultiplier (α / 2))) hsum
  have hscaled := mul_le_mul_of_nonneg_left hsq
    (by positivity : 0 ≤ K / ((n : ℝ) * reg))
  unfold separationRequirement
  apply lt_of_le_of_lt hscaled
  rw [div_mul_eq_mul_div]
  apply (div_lt_iff₀ (mul_pos hnR hreg)).2
  have hmul := (div_lt_iff₀ (mul_pos hreg hδ)).1 hsecond
  nlinarith

/-! ## Fixed-regularization asymptotics -/

open Filter
open scoped Topology

theorem meanRadius_tendsto_zero (K α : ℝ) :
    Tendsto (fun n : ℕ => meanRadius K n α) atTop (𝓝 0) := by
  have h : Tendsto (fun n : ℕ => 2 * K / (n : ℝ)) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop tendsto_natCast_atTop_atTop
  simpa only [meanRadius, Real.sqrt_zero, zero_mul] using
    h.sqrt.mul (tendsto_const_nhds (x := 1 + Real.sqrt (Real.log (1 / α))))

theorem operatorRadius_tendsto_zero (K α : ℝ) :
    Tendsto (fun n : ℕ => operatorRadius K n α) atTop (𝓝 0) := by
  have h : Tendsto (fun n : ℕ => K / Real.sqrt (n : ℝ)) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop (Real.tendsto_sqrt_atTop.comp tendsto_natCast_atTop_atTop)
  simpa only [operatorRadius, zero_mul] using
    h.mul (tendsto_const_nhds (x := 1 + Real.sqrt (Real.log (1 / α))))

theorem nullBound_tendsto_zero (K reg α : ℝ) :
    Tendsto (fun n : ℕ => nullBound K n reg α) atTop (𝓝 0) := by
  simpa only [nullBound, upperBound, zero_pow (by decide : 2 ≠ 0), zero_div] using
    ((meanRadius_tendsto_zero K α).pow 2).div_const (2 * reg)

theorem alternativeBound_tendsto_separation {δ : ℝ} (hδ : 0 ≤ δ) (K reg α : ℝ) :
    Tendsto (fun n : ℕ => alternativeBound δ K n reg α) atTop (𝓝 δ) := by
  have hm := (meanRadius_tendsto_zero K (α / 2)).div_const (Real.sqrt reg)
  have he := (operatorRadius_tendsto_zero K (α / 2)).div_const reg
  have ht := (((tendsto_const_nhds (x := Real.sqrt (2 * δ))).sub hm).max
    (tendsto_const_nhds (x := (0 : ℝ)))).pow 2
  have hd := (tendsto_const_nhds (x := (2 : ℝ))).mul
    ((tendsto_const_nhds (x := (1 : ℝ))).add he)
  have h := ht.div hd (by simp)
  have hfun : ((fun n : ℕ => max (Real.sqrt (2 * δ) - meanRadius K n (α / 2) /
      Real.sqrt reg) 0 ^ 2) / (fun n : ℕ => 2 * (1 + operatorRadius K n (α / 2) / reg))) =
      (fun n : ℕ => max (Real.sqrt (2 * δ) - meanRadius K n (α / 2) /
        Real.sqrt reg) 0 ^ 2 / (2 * (1 + operatorRadius K n (α / 2) / reg))) := by
    funext n
    exact Pi.div_apply _ _ _
  rw [hfun] at h
  simpa only [alternativeBound, lowerBound, zero_div, sub_zero, add_zero,
    max_eq_left (Real.sqrt_nonneg _), Real.sq_sqrt (by positivity : 0 ≤ 2 * δ),
    mul_one, Pi.div_apply, mul_div_cancel_left₀ _ (by norm_num : (2 : ℝ) ≠ 0)] using h

theorem eventually_nonempty_threshold {δ K reg α₀ α₁ : ℝ}
    (hδ : 0 < δ) : ∀ᶠ n : ℕ in atTop,
      ∃ η : ℝ, nullBound K n reg α₀ ≤ η ∧ η < alternativeBound δ K n reg α₁ := by
  have hgap := (nullBound_tendsto_zero K reg α₀).eventually_lt
    (alternativeBound_tendsto_separation hδ.le K reg α₁) hδ
  filter_upwards [hgap] with n hn
  exact ⟨nullBound K n reg α₀, le_rfl, hn⟩

end PearsonClassification
