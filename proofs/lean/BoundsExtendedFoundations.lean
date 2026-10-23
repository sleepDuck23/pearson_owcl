module

public import BoundsExtendedStatistical

/-! Remaining analytic foundations and the empirical variational problem. -/

@[expose] public section
noncomputable section
namespace PearsonClassification
open MeasureTheory InnerProductSpace
open scoped RealInnerProductSpace
variable {X E : Type*} [MeasurableSpace X]
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

namespace Moments

omit [InnerProductSpace ℝ E] in
/-- The manuscript's measurable separable-Hilbert-space assumptions give the
strong measurability needed for Bochner integration. -/
theorem feature_stronglyMeasurable [MeasurableSpace E] [BorelSpace E]
    [TopologicalSpace.SeparableSpace E] {φ : X → E} (hφ : Measurable φ) : StronglyMeasurable φ :=
  hφ.stronglyMeasurable

omit [MeasurableSpace X] [CompleteSpace E] in
theorem empiricalMean_inner {n : ℕ} (φ : X → E) (sample : Fin n → X) (f : E) :
    ⟪f, empiricalMean φ sample⟫ = (n : ℝ)⁻¹ * ∑ i, ⟪f, φ (sample i)⟫ := by
  simp only [empiricalMean, inner_smul_right, inner_sum]

omit [MeasurableSpace X] [CompleteSpace E] in
theorem empiricalSecondMoment_quadratic {n : ℕ} (φ : X → E)
    (sample : Fin n → X) (f : E) :
    ⟪f, empiricalSecondMoment φ sample f⟫ =
      (n : ℝ)⁻¹ * ∑ i, ⟪f, φ (sample i)⟫ ^ 2 := by
  simp only [empiricalSecondMoment, smul_apply, sum_apply, inner_smul_right,
    inner_sum, rankOne_apply]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  rw [real_inner_comm f (φ (sample i))]
  ring

omit [MeasurableSpace X] [CompleteSpace E] in
theorem empiricalSecondMoment_upper {n : ℕ} (φ : X → E) (sample : Fin n → X)
    {K : ℝ} (hn : 0 < n) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (f : E) :
    ⟪f, empiricalSecondMoment φ sample f⟫ ≤ K * ‖f‖ ^ 2 := by
  rw [empiricalSecondMoment_quadratic]
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast ne_of_gt hn
  have hsum : ∑ i, ⟪f, φ (sample i)⟫ ^ 2 ≤ ∑ _ : Fin n, K * ‖f‖ ^ 2 := by
    apply Finset.sum_le_sum
    intro i _
    have hcs := abs_real_inner_le_norm f (φ (sample i))
    have hs := (sq_le_sq₀ (abs_nonneg _) (by positivity)).2 hcs
    have hd := mul_le_mul_of_nonneg_left (hdiag (sample i)) (sq_nonneg ‖f‖)
    nlinarith [sq_abs ⟪f, φ (sample i)⟫]
  calc
    _ ≤ (n : ℝ)⁻¹ * ∑ _ : Fin n, K * ‖f‖ ^ 2 :=
      mul_le_mul_of_nonneg_left hsum (by positivity)
    _ = K * ‖f‖ ^ 2 := by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      field_simp

omit [MeasurableSpace X] [CompleteSpace E] in
theorem empirical_variational_objective {n : ℕ} (φ : X → E)
    (reference incoming : Fin n → X) (reg : ℝ) (f : E) :
    Operators.objective (Operators.regularized (empiricalSecondMoment φ reference) reg)
      (empiricalMean φ incoming - empiricalMean φ reference) f =
      (n : ℝ)⁻¹ * ∑ i, ⟪f, φ (incoming i)⟫ -
      (n : ℝ)⁻¹ * ∑ i, ⟪f, φ (reference i)⟫ -
      ((n : ℝ)⁻¹ * ∑ i, ⟪f, φ (reference i)⟫ ^ 2) / 2 - reg / 2 * ‖f‖ ^ 2 := by
  simp only [Operators.objective, Operators.regularized, add_apply, smul_apply,
    ContinuousLinearMap.id_apply, inner_sub_right, inner_add_right, inner_smul_right,
    real_inner_self_eq_norm_sq, empiricalMean_inner, empiricalSecondMoment_quadratic]
  ring

omit [MeasurableSpace X] in
theorem empiricalScore_variational {n : ℕ} (φ : X → E)
    (reference incoming : Fin n → X) (reg : ℝ) (hreg : 0 < reg) :
    sSup (Set.range (fun f : E =>
      (n : ℝ)⁻¹ * ∑ i, ⟪f, φ (incoming i)⟫ -
      (n : ℝ)⁻¹ * ∑ i, ⟪f, φ (reference i)⟫ -
      ((n : ℝ)⁻¹ * ∑ i, ⟪f, φ (reference i)⟫ ^ 2) / 2 - reg / 2 * ‖f‖ ^ 2)) =
      empiricalScore φ reference incoming reg hreg := by
  let A := Operators.regularizedEquiv (empiricalSecondMoment φ reference)
    (empiricalSecondMoment_positive φ reference) reg hreg
  have hA : A.toContinuousLinearMap =
      Operators.regularized (empiricalSecondMoment φ reference) reg :=
    Operators.regularizedEquiv_coe _ _ _ _
  have hpos : A.toContinuousLinearMap.IsPositive := by
    rw [hA]
    exact Operators.regularized_positive _ (empiricalSecondMoment_positive φ reference) hreg.le
  have hid : (fun f : E =>
      (n : ℝ)⁻¹ * ∑ i, ⟪f, φ (incoming i)⟫ -
      (n : ℝ)⁻¹ * ∑ i, ⟪f, φ (reference i)⟫ -
      ((n : ℝ)⁻¹ * ∑ i, ⟪f, φ (reference i)⟫ ^ 2) / 2 - reg / 2 * ‖f‖ ^ 2) =
      Operators.objective A.toContinuousLinearMap
        (empiricalMean φ incoming - empiricalMean φ reference) := by
    funext f
    rw [hA, empirical_variational_objective]
  rw [hid]
  exact Operators.variational_supremum A hpos _

omit [MeasurableSpace X] in
theorem empiricalScore_unique_maximizer {n : ℕ} (φ : X → E)
    (reference incoming : Fin n → X) (reg : ℝ) (hreg : 0 < reg) (f : E) :
    Operators.objective (Operators.regularized (empiricalSecondMoment φ reference) reg)
      (empiricalMean φ incoming - empiricalMean φ reference) f =
      empiricalScore φ reference incoming reg hreg ↔
    f = (Operators.regularizedEquiv (empiricalSecondMoment φ reference)
      (empiricalSecondMoment_positive φ reference) reg hreg).symm
        (empiricalMean φ incoming - empiricalMean φ reference) := by
  let A := Operators.regularizedEquiv (empiricalSecondMoment φ reference)
    (empiricalSecondMoment_positive φ reference) reg hreg
  have hA : A.toContinuousLinearMap =
      Operators.regularized (empiricalSecondMoment φ reference) reg :=
    Operators.regularizedEquiv_coe _ _ _ _
  have hp : A.toContinuousLinearMap.IsPositive := by
    rw [hA]
    exact Operators.regularized_positive _ (empiricalSecondMoment_positive φ reference) hreg.le
  have hc (g : E) : reg * ‖g‖ ^ 2 ≤ ⟪g, A g⟫ := by
    change reg * ‖g‖ ^ 2 ≤ ⟪g, A.toContinuousLinearMap g⟫
    rw [hA]
    exact Operators.regularized_coercive _ (empiricalSecondMoment_positive φ reference) reg g
  simpa only [hA, A, empiricalScore] using Operators.objective_eq_score_iff A hp hreg hc
    (empiricalMean φ incoming - empiricalMean φ reference) f

end Moments
end PearsonClassification
