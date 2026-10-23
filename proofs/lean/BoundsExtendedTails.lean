module

public import BoundsExtendedAppendices

/-! Explicit exponential tails and tensor-operator variance from the main text. -/

@[expose] public section
noncomputable section
namespace PearsonClassification.Statistical
open MeasureTheory BoundedDifferences Sampling
open scoped RealInnerProductSpace
variable {X E : Type*} [MeasurableSpace X]
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

def meanError {n : ℕ} (P Q : Measure X) (φ : X → E) (s : Sample X (n + n)) : ℝ :=
  ‖empiricalDelta φ s - (Moments.mean Q φ - Moments.mean P φ)‖

def tensorError {n : ℕ} (P : Measure X) (φ : X → E) (s : Sample X (n + n)) : ℝ :=
  ‖Moments.empiricalMean (fun x => HilbertTensor.liftRankOne (φ x)) (reference s) -
    Moments.mean P (fun x => HilbertTensor.liftRankOne (φ x))‖

omit [CompleteSpace E] in
theorem mean_tail (n : ℕ) (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {K t : ℝ} (hφ : StronglyMeasurable φ) (hK : 0 < K)
    (hn : 0 < n) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K)
    (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫) (ht : 0 ≤ t) :
    (batchLaw n P Q).real {s | t ≤ meanError P Q φ s -
      ∫ q, meanError P Q φ q ∂batchLaw n P Q} ≤
      Real.exp (-(n : ℝ) * t ^ 2 / (2 * K)) := by
  have h := weighted_mean_tail (n + n) (batchMeasures n P Q)
    (blockWeights n (-(n : ℝ)⁻¹) ((n : ℝ)⁻¹)) hφ hK.le hdiag hkernel ht
  simp_rw [mean_error_identity P Q φ hn] at h
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast ne_of_gt hn
  have hv : K * ∑ i, blockWeights n (-(n : ℝ)⁻¹) ((n : ℝ)⁻¹) i ^ 2 =
      2 * K / (n : ℝ) := by rw [blockWeights_square_sum]; field_simp; ring
  rw [hv] at h
  have hexp : -t ^ 2 / (2 * K / (n : ℝ)) = -(n : ℝ) * t ^ 2 / (2 * K) := by
    field_simp
  rw [hexp] at h
  exact h

omit [CompleteSpace E] in
theorem operator_tensor_tail (n : ℕ) (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {K t : ℝ} (hφ : StronglyMeasurable φ) (hK : 0 < K)
    (hn : 0 < n) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (ht : 0 ≤ t) :
    (batchLaw n P Q).real {s | t ≤ tensorError P φ s -
      ∫ q, tensorError P φ q ∂batchLaw n P Q} ≤
      Real.exp (-(n : ℝ) * t ^ 2 / K ^ 2) := by
  have h := weighted_mean_tail (n + n) (batchMeasures n P Q)
    (blockWeights n ((n : ℝ)⁻¹) 0) (tensorFeature_measurable hφ) (sq_nonneg K)
    (tensorFeature_diagonal hK.le hdiag)
    (fun x y => by rw [HilbertTensor.inner_liftRankOne]; exact sq_nonneg _) ht
  simp_rw [reference_error_identity P Q _ hn] at h
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast ne_of_gt hn
  have hv : K ^ 2 * ∑ i, blockWeights n ((n : ℝ)⁻¹) 0 i ^ 2 =
      K ^ 2 / (n : ℝ) := by rw [blockWeights_square_sum]; field_simp; ring
  rw [hv] at h
  have hexp : -t ^ 2 / (K ^ 2 / (n : ℝ)) = -(n : ℝ) * t ^ 2 / K ^ 2 := by field_simp
  rw [hexp] at h
  exact h

omit [CompleteSpace E] in
theorem operator_tensor_variance (n : ℕ) (P : Measure X) [IsProbabilityMeasure P]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ) (hK : 0 ≤ K)
    (hn : 0 < n) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) :
    (∫ s, ‖Moments.empiricalMean (fun x => HilbertTensor.liftRankOne (φ x))
      (observations n s) - Moments.mean P (fun x => HilbertTensor.liftRankOne (φ x))‖ ^ 2
        ∂law n (fun _ => P)) =
    ((∫ x, ‖φ x‖ ^ 4 ∂P) -
      ‖Moments.mean P (fun x => HilbertTensor.liftRankOne (φ x))‖ ^ 2) / (n : ℝ) := by
  have h := iid_mean_variance n P (tensorFeature_measurable hφ) hn
    (tensorFeature_diagonal hK hdiag)
  simpa only [HilbertTensor.norm_liftRankOne, ← pow_mul] using h

theorem mean_high_probability (n : ℕ) (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {K γ : ℝ} (hφ : StronglyMeasurable φ) (hK : 0 < K)
    (hn : 0 < n) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K)
    (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫) (hγ : 0 < γ) (hγ₁ : γ < 1) :
    1 - ENNReal.ofReal γ ≤ batchLaw n P Q {s | meanError P Q φ s ≤ meanRadius K n γ} := by
  apply probability_lower_of_complement
  simpa only [Set.compl_ofPred, not_le, meanError, empiricalDelta] using
    mean_concentration n P Q hφ hK hn hdiag hkernel hγ hγ₁

theorem operator_high_probability (n : ℕ) (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {K γ : ℝ} (hφ : StronglyMeasurable φ) (hK : 0 < K)
    (hn : 0 < n) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hγ : 0 < γ) (hγ₁ : γ < 1) :
    1 - ENNReal.ofReal γ ≤ batchLaw n P Q {s |
      ‖Moments.empiricalSecondMoment φ (reference s) - Moments.secondMoment P φ‖ ≤
        operatorRadius K n γ} := by
  apply probability_lower_of_complement
  simpa only [Set.compl_ofPred, not_le, meanError, empiricalDelta] using
    operator_concentration n P Q hφ hK hn hdiag hγ hγ₁

end PearsonClassification.Statistical
