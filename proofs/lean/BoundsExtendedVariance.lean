module

public import BoundsExtendedStatistical

/-! Exact variance statements underlying the concentration proof. -/

@[expose] public section
noncomputable section
namespace PearsonClassification
open MeasureTheory BoundedDifferences Sampling Statistical
open scoped RealInnerProductSpace
variable {X E : Type*} [MeasurableSpace X]
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

namespace Moments

/-- The variance identity for arbitrary integrable, square-integrable features;
it does not require a pointwise bound or a zero uncentered mean. -/
theorem variance_identity (μ : Measure X) [IsProbabilityMeasure μ]
    {φ : X → E} (hi : Integrable φ μ) (hsq : Integrable (fun x => ‖φ x‖ ^ 2) μ) :
    (∫ x, ‖φ x - mean μ φ‖ ^ 2 ∂μ) =
      (∫ x, ‖φ x‖ ^ 2 ∂μ) - ‖mean μ φ‖ ^ 2 := by
  have hinner := (hi.const_inner (mean μ φ)).const_mul (2 : ℝ)
  have hsub : Integrable (fun x => ‖φ x‖ ^ 2 - 2 * ⟪mean μ φ, φ x⟫) μ := hsq.sub' hinner
  have hexpand : (fun x => ‖φ x - mean μ φ‖ ^ 2) =
      (fun x => ‖φ x‖ ^ 2 - 2 * ⟪mean μ φ, φ x⟫ + ‖mean μ φ‖ ^ 2) := by
    funext x
    rw [norm_sub_sq_real, real_inner_comm (mean μ φ) (φ x)]
  rw [hexpand, integral_add hsub (integrable_const _),
    integral_sub hsq hinner, integral_const_mul, integral_inner hi]
  change (∫ x, ‖φ x‖ ^ 2 ∂μ) - 2 * ⟪mean μ φ, mean μ φ⟫ +
    (∫ _ : X, ‖mean μ φ‖ ^ 2 ∂μ) = _
  simp only [real_inner_self_eq_norm_sq, integral_const, probReal_univ, one_smul]
  ring

end Moments

namespace Sampling

theorem centeredSum_variance_exact (n : ℕ) (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)] (w : Fin n → ℝ) {φ : X → E} {K : ℝ}
    (hφ : StronglyMeasurable φ) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) :
    (∫ s, ‖sumStat (centeredFamily μ φ w) s‖ ^ 2 ∂law n μ) =
      ∑ i, w i ^ 2 * ((∫ x, ‖φ x‖ ^ 2 ∂μ i) - ‖Moments.mean (μ i) φ‖ ^ 2) := by
  rw [sumStat_variance n μ _ (centeredFamily_measurable μ w hφ)
    (familyBound μ φ w K) (centeredFamily_bound μ φ w hdiag)
    (centeredFamily_center μ w hφ hdiag)]
  apply Finset.sum_congr rfl
  intro i _
  simp only [centeredFamily, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
  rw [integral_const_mul, Moments.feature_variance_identity (μ i) hφ hdiag]

end Sampling

namespace Statistical

theorem mean_error_second_moment (n : ℕ) (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hn : 0 < n) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) :
    (∫ s, ‖(Moments.empiricalMean φ (incoming s) -
      Moments.empiricalMean φ (reference s)) -
      (Moments.mean Q φ - Moments.mean P φ)‖ ^ 2 ∂batchLaw n P Q) =
    ((∫ x, ‖φ x‖ ^ 2 ∂P) - ‖Moments.mean P φ‖ ^ 2 +
      ((∫ x, ‖φ x‖ ^ 2 ∂Q) - ‖Moments.mean Q φ‖ ^ 2)) / (n : ℝ) := by
  have h := centeredSum_variance_exact (n + n) (batchMeasures n P Q)
    (blockWeights n (-(n : ℝ)⁻¹) ((n : ℝ)⁻¹)) hφ hdiag
  simp_rw [mean_error_identity P Q φ hn] at h
  change (∫ s, ‖(Moments.empiricalMean φ (incoming s) -
    Moments.empiricalMean φ (reference s)) -
    (Moments.mean Q φ - Moments.mean P φ)‖ ^ 2 ∂batchLaw n P Q) = _ at h
  rw [h, Fin.sum_univ_add]
  simp only [batchMeasures, blockWeights, Fin.addCases_left, Fin.addCases_right,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast ne_of_gt hn
  field_simp

end Statistical
end PearsonClassification
