module

public import BoundsExtendedVariance
public import BoundsExtendedExamples

/-! Additional direct correspondences to the manuscript appendices. -/

@[expose] public section
noncomputable section
namespace PearsonClassification
open MeasureTheory InnerProductSpace BoundedDifferences Sampling Statistical
open scoped RealInnerProductSpace TensorProduct
variable {X E : Type*} [MeasurableSpace X]
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

namespace Moments

omit [MeasurableSpace X] in
theorem rkhsFeature_inner [RKHS ℝ E X ℝ] (x y : X) :
    ⟪rkhsFeature (E := E) x, rkhsFeature (E := E) y⟫ = RKHS.kernel E x y 1 := by
  simp only [rkhsFeature, RKHS.kerFun_inner, RKHS.kerFun_apply, Real.inner_apply, one_mul]

omit [MeasurableSpace X] in
theorem rkhsFeature_norm_sq [RKHS ℝ E X ℝ] (x : X) :
    ‖rkhsFeature (E := E) x‖ ^ 2 = RKHS.kernel E x x 1 := by
  rw [← real_inner_self_eq_norm_sq, rkhsFeature_inner]

omit [MeasurableSpace X] [CompleteSpace E] in
theorem kernel_uniform_bound {φ : X → E} {K : ℝ} (hK : 0 ≤ K)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (x y : X) : |⟪φ x, φ y⟫| ≤ K := by
  calc
    |⟪φ x, φ y⟫| ≤ ‖φ x‖ * ‖φ y‖ := abs_real_inner_le_norm _ _
    _ ≤ Real.sqrt K * Real.sqrt K :=
      mul_le_mul (feature_norm_bound hdiag x) (feature_norm_bound hdiag y)
        (norm_nonneg _) (Real.sqrt_nonneg _)
    _ = K := Real.mul_self_sqrt hK

omit [MeasurableSpace X] [CompleteSpace E] in
theorem evaluation_uniform_bound {φ : X → E} {K : ℝ}
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (f : E) (x : X) :
    |⟪f, φ x⟫| ≤ Real.sqrt K * ‖f‖ := by
  calc
    |⟪f, φ x⟫| ≤ ‖f‖ * ‖φ x‖ := abs_real_inner_le_norm _ _
    _ ≤ ‖f‖ * Real.sqrt K := mul_le_mul_of_nonneg_left
      (feature_norm_bound hdiag x) (norm_nonneg _)
    _ = Real.sqrt K * ‖f‖ := mul_comm _ _

theorem populationScore_zero_iff (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {K reg : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hreg : 0 < reg) :
    populationScore P Q hφ hdiag reg hreg = 0 ↔ mean Q φ = mean P φ := by
  exact (Operators.quadraticScore_eq_zero_iff (populationOperator P hφ hdiag reg hreg)
    hreg (population_coercive P hφ hdiag hreg) _).trans sub_eq_zero

theorem population_unique_maximizer (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {K reg : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hreg : 0 < reg) (f : E) :
    Operators.objective (populationOperator P hφ hdiag reg hreg).toContinuousLinearMap
      (mean Q φ - mean P φ) f = populationScore P Q hφ hdiag reg hreg ↔
    f = (populationOperator P hφ hdiag reg hreg).symm (mean Q φ - mean P φ) :=
  Operators.objective_eq_score_iff _ (population_positive P hφ hdiag hreg) hreg
    (population_coercive P hφ hdiag hreg) _ f

end Moments

namespace Operators
omit [MeasurableSpace X] [CompleteSpace E] in
theorem inverse_operator_norm_bound (A : E ≃L[ℝ] E) {reg : ℝ} (hreg : 0 < reg)
    (hcoercive : ∀ f, reg * ‖f‖ ^ 2 ≤ ⟪f, A f⟫) :
    ‖A.symm.toContinuousLinearMap‖ ≤ 1 / reg := by
  apply ContinuousLinearMap.opNorm_le_bound _ (by positivity)
  intro v
  simpa only [one_div, div_eq_inv_mul, mul_one, ContinuousLinearEquiv.coe_coe]
    using inverse_norm_bound A hreg hcoercive v
end Operators

namespace HilbertTensor
omit [MeasurableSpace X] [CompleteSpace E] in
theorem norm_completed_rankOne (u v : E) :
    ‖((u ⊗ₜ[ℝ] v : E ⊗[ℝ] E) : Space E)‖ = ‖u‖ * ‖v‖ := by simp

omit [MeasurableSpace X] [CompleteSpace E] in
theorem inner_completed_rankOne (u v a b : E) :
    ⟪((u ⊗ₜ[ℝ] v : E ⊗[ℝ] E) : Space E),
      ((a ⊗ₜ[ℝ] b : E ⊗[ℝ] E) : Space E)⟫ = ⟪u, a⟫ * ⟪v, b⟫ := by simp
end HilbertTensor

namespace Sampling

theorem iid_mean_variance (n : ℕ) (P : Measure X) [IsProbabilityMeasure P]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hn : 0 < n) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) :
    (∫ s, ‖Moments.empiricalMean φ (observations n s) - Moments.mean P φ‖ ^ 2
      ∂law n (fun _ => P)) =
      ((∫ x, ‖φ x‖ ^ 2 ∂P) - ‖Moments.mean P φ‖ ^ 2) / (n : ℝ) := by
  have hid (s : Sample X n) :
      sumStat (centeredFamily (fun _ : Fin n => P) φ (fun _ => (n : ℝ)⁻¹)) s =
      Moments.empiricalMean φ (observations n s) - Moments.mean P φ := by
    simp only [sumStat, centeredFamily, ← Finset.smul_sum]
    exact empirical_centering φ (observations n s) (Moments.mean P φ) hn
  have h := centeredSum_variance_exact n (fun _ => P) (fun _ => (n : ℝ)⁻¹) hφ hdiag
  simp_rw [hid] at h
  rw [h]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast ne_of_gt hn
  field_simp

end Sampling

omit [CompleteSpace E] in
/-- Mixtures approach the reference in score, for every kernel satisfying the
feature assumptions. No characteristic-kernel hypothesis is needed. -/
theorem mixture_score_tendsto_zero (P R : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure R] (A : E ≃L[ℝ] E)
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) :
    Filter.Tendsto (fun ε => Operators.quadraticScore A
      (Moments.mean (Moments.mixture P R ε) φ - Moments.mean P φ))
      (nhdsWithin 0 (Set.Icc 0 1)) (nhds 0) := by
  let D := Operators.quadraticScore A (Moments.mean R φ - Moments.mean P φ)
  have hc : Continuous (fun ε : ℝ => ε ^ 2 * D) := by fun_prop
  have ht : Filter.Tendsto (fun ε : ℝ => ε ^ 2 * D)
      (nhdsWithin 0 (Set.Icc 0 1)) (nhds (0 ^ 2 * D)) := hc.continuousWithinAt
  simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_mul] at ht
  apply ht.congr'
  filter_upwards [self_mem_nhdsWithin] with ε hε
  exact (Moments.mixture_score_scaling P R A hφ hdiag hε.1 hε.2).symm

end PearsonClassification
