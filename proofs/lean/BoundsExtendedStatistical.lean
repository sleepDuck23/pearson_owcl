module

public import BoundsExtendedSampling
public import BoundsExtendedProbability

/-! Main statistical results on the joint independent batch-sampling law. -/

@[expose] public section

noncomputable section

namespace PearsonClassification
namespace Statistical

open MeasureTheory InnerProductSpace BoundedDifferences Sampling
open scoped RealInnerProductSpace NNReal TensorProduct

variable {X E : Type*} [MeasurableSpace X]
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

def batchMeasures (n : ℕ) (P Q : Measure X) : Fin (n + n) → Measure X :=
  Fin.addCases (fun _ => P) (fun _ => Q)

theorem batchMeasures_probability (n : ℕ) (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q] (i : Fin (n + n)) :
    IsProbabilityMeasure (batchMeasures n P Q i) := by
  refine Fin.addCases (fun j => ?_) (fun j => ?_) i
  · simpa only [batchMeasures, Fin.addCases_left] using (inferInstance : IsProbabilityMeasure P)
  · simpa only [batchMeasures, Fin.addCases_right] using (inferInstance : IsProbabilityMeasure Q)

instance (n : ℕ) (P Q : Measure X) [IsProbabilityMeasure P] [IsProbabilityMeasure Q] :
    (i : Fin (n + n)) → IsProbabilityMeasure (batchMeasures n P Q i) :=
  batchMeasures_probability n P Q

def batchLaw (n : ℕ) (P Q : Measure X) : Measure (Sample X (n + n)) :=
  law (n + n) (batchMeasures n P Q)

instance (n : ℕ) (P Q : Measure X) [IsProbabilityMeasure P] [IsProbabilityMeasure Q] :
    IsProbabilityMeasure (batchLaw n P Q) :=
  law_isProbability _ _ (batchMeasures_probability n P Q)

def reference {n : ℕ} (s : Sample X (n + n)) (i : Fin n) : X :=
  observations (n + n) s (Fin.castAdd n i)

def incoming {n : ℕ} (s : Sample X (n + n)) (i : Fin n) : X :=
  observations (n + n) s (Fin.natAdd n i)

def blockWeights (n : ℕ) (a b : ℝ) : Fin (n + n) → ℝ :=
  Fin.addCases (fun _ => a) (fun _ => b)

theorem blockWeights_square_sum (n : ℕ) (a b : ℝ) :
    (∑ i, blockWeights n a b i ^ 2) = (n : ℝ) * (a ^ 2 + b ^ 2) := by
  rw [Fin.sum_univ_add]
  simp only [blockWeights, Fin.addCases_left, Fin.addCases_right, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

omit [MeasurableSpace X] [CompleteSpace E] in
theorem empirical_centering {n : ℕ} (φ : X → E) (sample : Fin n → X)
    (m : E) (hn : 0 < n) :
    (n : ℝ)⁻¹ • ∑ i, (φ (sample i) - m) = Moments.empiricalMean φ sample - m := by
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast ne_of_gt hn
  rw [Finset.sum_sub_distrib, smul_sub]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin]
  rw [← Nat.cast_smul_eq_nsmul ℝ, smul_smul, inv_mul_cancel₀ hnR, one_smul]
  rfl

omit [CompleteSpace E] in
theorem mean_error_identity {n : ℕ} (P Q : Measure X) (φ : X → E)
    (hn : 0 < n) (s : Sample X (n + n)) :
    sumStat (centeredFamily (batchMeasures n P Q) φ
      (blockWeights n (-(n : ℝ)⁻¹) ((n : ℝ)⁻¹))) s =
      (Moments.empiricalMean φ (incoming s) - Moments.empiricalMean φ (reference s)) -
        (Moments.mean Q φ - Moments.mean P φ) := by
  have href := empirical_centering φ (reference s) (Moments.mean P φ) hn
  have hin := empirical_centering φ (incoming s) (Moments.mean Q φ) hn
  simp only [reference] at href
  simp only [incoming] at hin
  simp only [sumStat, Fin.sum_univ_add, centeredFamily, batchMeasures, blockWeights,
    Fin.addCases_left, Fin.addCases_right]
  rw [← Finset.smul_sum, ← Finset.smul_sum]
  rw [neg_smul, href, hin]
  abel

omit [CompleteSpace E] in
theorem reference_error_identity {n : ℕ} (P Q : Measure X) (φ : X → E)
    (hn : 0 < n) (s : Sample X (n + n)) :
    sumStat (centeredFamily (batchMeasures n P Q) φ
      (blockWeights n ((n : ℝ)⁻¹) 0)) s =
      Moments.empiricalMean φ (reference s) - Moments.mean P φ := by
  simp only [sumStat, Fin.sum_univ_add, centeredFamily, batchMeasures, blockWeights,
    Fin.addCases_left, Fin.addCases_right, zero_smul, Finset.sum_const_zero, add_zero]
  rw [← Finset.smul_sum]
  exact empirical_centering φ (reference s) (Moments.mean P φ) hn

/-- Mean concentration under the manuscript's joint independent sampling law. -/
theorem mean_concentration (n : ℕ) (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {K γ : ℝ} (hφ : StronglyMeasurable φ) (hK : 0 < K)
    (hn : 0 < n) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K)
    (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫) (hγ : 0 < γ) (hγ₁ : γ < 1) :
    batchLaw n P Q {s | meanRadius K n γ <
      ‖(Moments.empiricalMean φ (incoming s) - Moments.empiricalMean φ (reference s)) -
        (Moments.mean Q φ - Moments.mean P φ)‖} ≤ ENNReal.ofReal γ := by
  let w := blockWeights n (-(n : ℝ)⁻¹) ((n : ℝ)⁻¹)
  have hnR : 0 < (n : ℝ) := by exact_mod_cast hn
  have hvar : K * ∑ i, w i ^ 2 = 2 * K / (n : ℝ) := by
    rw [blockWeights_square_sum]
    field_simp
    ring
  have hV : 0 < K * ∑ i, w i ^ 2 := by rw [hvar]; positivity
  have h := weighted_mean_concentration (n + n) (batchMeasures n P Q) w
    hφ hK.le hdiag hkernel hV hγ hγ₁
  dsimp only [w] at h
  dsimp only [w] at hvar
  simp_rw [mean_error_identity P Q φ hn] at h
  simpa only [batchLaw, meanRadius, hvar] using h

omit [CompleteSpace E] in
theorem tensorFeature_measurable {φ : X → E} (hφ : StronglyMeasurable φ) :
    StronglyMeasurable (fun x => HilbertTensor.liftRankOne (φ x)) := by
  have hc : Continuous (fun v : E => HilbertTensor.liftRankOne v) := by
    change Continuous (fun v : E =>
      (UniformSpace.Completion.toComplL : (E ⊗[ℝ] E) →L[ℝ] HilbertTensor.Space E)
        (TensorProduct.mkL ℝ E E v v))
    exact UniformSpace.Completion.toComplL.continuous.comp
      ((TensorProduct.mkL ℝ E E).continuous.clm_apply continuous_id)
  exact hc.comp_stronglyMeasurable hφ

omit [MeasurableSpace X] [CompleteSpace E] in
theorem tensorFeature_diagonal {φ : X → E} {K : ℝ} (hK : 0 ≤ K)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (x : X) :
    ‖HilbertTensor.liftRankOne (φ x)‖ ^ 2 ≤ K ^ 2 := by
  rw [HilbertTensor.norm_liftRankOne]
  exact (sq_le_sq₀ (sq_nonneg _) hK).2 (hdiag x)

theorem tensorMean_toOperator (P : Measure X) [IsProbabilityMeasure P]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) :
    HilbertTensor.toOperator (Moments.mean P (fun x => HilbertTensor.liftRankOne (φ x))) =
      Moments.secondMoment P φ := by
  have hi : Integrable (fun x => HilbertTensor.liftRankOne (φ x)) P :=
    Integrable.of_bound (tensorFeature_measurable hφ).aestronglyMeasurable K
      (ae_of_all P fun x => by simpa only [HilbertTensor.norm_liftRankOne] using hdiag x)
  rw [Moments.mean, ← HilbertTensor.toOperator.integral_comp_comm hi]
  simp only [HilbertTensor.toOperator_liftRankOne, Moments.secondMoment]

omit [MeasurableSpace X] in
theorem tensorEmpirical_toOperator {n : ℕ} (φ : X → E) (sample : Fin n → X) :
    HilbertTensor.toOperator (Moments.empiricalMean
      (fun x => HilbertTensor.liftRankOne (φ x)) sample) =
        Moments.empiricalSecondMoment φ sample := by
  simp only [Moments.empiricalMean, Moments.empiricalSecondMoment, map_smul, map_sum,
    HilbertTensor.toOperator_liftRankOne]

theorem operator_error_le_tensor (P : Measure X) [IsProbabilityMeasure P]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) {n : ℕ} (sample : Fin n → X) :
    ‖Moments.empiricalSecondMoment φ sample - Moments.secondMoment P φ‖ ≤
      ‖Moments.empiricalMean (fun x => HilbertTensor.liftRankOne (φ x)) sample -
        Moments.mean P (fun x => HilbertTensor.liftRankOne (φ x))‖ := by
  rw [← tensorEmpirical_toOperator, ← tensorMean_toOperator P hφ hdiag]
  exact HilbertTensor.error_domination _ _

/-- Operator concentration on the same joint sampling law as the mean bound. -/
theorem operator_concentration (n : ℕ) (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {K γ : ℝ} (hφ : StronglyMeasurable φ) (hK : 0 < K)
    (hn : 0 < n) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hγ : 0 < γ) (hγ₁ : γ < 1) :
    batchLaw n P Q {s | operatorRadius K n γ <
      ‖Moments.empiricalSecondMoment φ (reference s) - Moments.secondMoment P φ‖} ≤
      ENNReal.ofReal γ := by
  let w := blockWeights n ((n : ℝ)⁻¹) 0
  have hnR : 0 < (n : ℝ) := by exact_mod_cast hn
  have hvar : K ^ 2 * ∑ i, w i ^ 2 = K ^ 2 / (n : ℝ) := by
    rw [blockWeights_square_sum]
    field_simp
    ring
  have hV : 0 < K ^ 2 * ∑ i, w i ^ 2 := by rw [hvar]; positivity
  have h := weighted_mean_concentration (n + n) (batchMeasures n P Q) w
    (tensorFeature_measurable hφ) (sq_nonneg K) (tensorFeature_diagonal hK.le hdiag)
    (fun x y => by rw [HilbertTensor.inner_liftRankOne]; exact sq_nonneg _) hV hγ hγ₁
  dsimp only [w] at h
  dsimp only [w] at hvar
  simp_rw [reference_error_identity P Q _ hn] at h
  have hrad : Real.sqrt (K ^ 2 / (n : ℝ)) = K / Real.sqrt (n : ℝ) := by
    rw [Real.sqrt_div (sq_nonneg K), Real.sqrt_sq hK.le]
  rw [hvar, hrad] at h
  apply (measure_mono ?_).trans h
  intro s hs
  change operatorRadius K n γ < _ at hs
  exact lt_of_lt_of_le hs (operator_error_le_tensor P hφ hdiag (reference s))

def populationOperator (P : Measure X) [IsProbabilityMeasure P]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (reg : ℝ) (hreg : 0 < reg) : E ≃L[ℝ] E :=
  Operators.regularizedEquiv (Moments.secondMoment P φ)
    (Moments.secondMoment_positive P hφ hdiag) reg hreg

def empiricalOperator {n : ℕ} (φ : X → E) (reg : ℝ) (hreg : 0 < reg)
    (s : Sample X (n + n)) : E ≃L[ℝ] E :=
  Operators.regularizedEquiv (Moments.empiricalSecondMoment φ (reference s))
    (Moments.empiricalSecondMoment_positive φ (reference s)) reg hreg

def empiricalDelta {n : ℕ} (φ : X → E) (s : Sample X (n + n)) : E :=
  Moments.empiricalMean φ (incoming s) - Moments.empiricalMean φ (reference s)

def batchScore {n : ℕ} (φ : X → E) (reg : ℝ) (hreg : 0 < reg)
    (s : Sample X (n + n)) : ℝ :=
  Moments.empiricalScore φ (reference s) (incoming s) reg hreg

theorem population_positive (P : Measure X) [IsProbabilityMeasure P]
    {φ : X → E} {K reg : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hreg : 0 < reg) :
    (populationOperator P hφ hdiag reg hreg).toContinuousLinearMap.IsPositive := by
  rw [populationOperator, Operators.regularizedEquiv_coe]
  exact Operators.regularized_positive _ (Moments.secondMoment_positive P hφ hdiag) hreg.le

theorem population_coercive (P : Measure X) [IsProbabilityMeasure P]
    {φ : X → E} {K reg : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hreg : 0 < reg) (f : E) :
    reg * ‖f‖ ^ 2 ≤ ⟪f, populationOperator P hφ hdiag reg hreg f⟫ := by
  change reg * ‖f‖ ^ 2 ≤ ⟪f, (populationOperator P hφ hdiag reg hreg).toContinuousLinearMap f⟫
  rw [populationOperator, Operators.regularizedEquiv_coe]
  exact Operators.regularized_coercive _ (Moments.secondMoment_positive P hφ hdiag) reg f

omit [MeasurableSpace X] in
theorem empirical_positive {n : ℕ} (φ : X → E) {reg : ℝ} (hreg : 0 < reg)
    (s : Sample X (n + n)) : (empiricalOperator φ reg hreg s).toContinuousLinearMap.IsPositive := by
  rw [empiricalOperator, Operators.regularizedEquiv_coe]
  exact Operators.regularized_positive _ (Moments.empiricalSecondMoment_positive φ _) hreg.le

omit [MeasurableSpace X] in
theorem empirical_coercive {n : ℕ} (φ : X → E) {reg : ℝ} (hreg : 0 < reg)
    (s : Sample X (n + n)) (f : E) : reg * ‖f‖ ^ 2 ≤ ⟪f, empiricalOperator φ reg hreg s f⟫ := by
  change reg * ‖f‖ ^ 2 ≤ ⟪f, (empiricalOperator φ reg hreg s).toContinuousLinearMap f⟫
  rw [empiricalOperator, Operators.regularizedEquiv_coe]
  exact Operators.regularized_coercive _ (Moments.empiricalSecondMoment_positive φ _) reg f

theorem operator_difference (P : Measure X) [IsProbabilityMeasure P]
    {φ : X → E} {K reg : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hreg : 0 < reg) {n : ℕ} (s : Sample X (n + n)) :
    (empiricalOperator φ reg hreg s).toContinuousLinearMap -
        (populationOperator P hφ hdiag reg hreg).toContinuousLinearMap =
      Moments.empiricalSecondMoment φ (reference s) - Moments.secondMoment P φ := by
  simp only [empiricalOperator, populationOperator, Operators.regularizedEquiv_coe,
    Operators.regularized]
  abel

/-- The null error theorem from the original kernel and sampling hypotheses;
there is no concentration or inverse-comparison premise. -/
theorem null_error (n : ℕ) (P : Measure X) [IsProbabilityMeasure P]
    {φ : X → E} {K reg α η : ℝ} (hφ : StronglyMeasurable φ)
    (hK : 0 < K) (hn : 0 < n) (hreg : 0 < reg)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫)
    (hα : 0 < α) (hα₁ : α < 1) (hthreshold : nullBound K n reg α ≤ η) :
    batchLaw n P P {s | η < batchScore φ reg hreg s} ≤ ENNReal.ofReal α := by
  exact null_error_of_concentration (batchLaw n P P) (empiricalOperator φ reg hreg)
    (Moments.mean P φ - Moments.mean P φ) (empiricalDelta φ) hreg (sub_self _)
    (empirical_coercive φ hreg)
    (mean_concentration n P P hφ hK hn hdiag hkernel hα hα₁) hthreshold

/-- The alternative error theorem, with both concentration bounds discharged. -/
theorem alternative_error (n : ℕ) (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {δ K reg α η : ℝ} (hφ : StronglyMeasurable φ)
    (hK : 0 < K) (hn : 0 < n) (hreg : 0 < reg) (hδ : 0 ≤ δ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫)
    (hα : 0 < α) (hα₁ : α < 1)
    (hseparation : δ ≤ Moments.populationScore P Q hφ hdiag reg hreg)
    (hthreshold : η < alternativeBound δ K n reg α) :
    batchLaw n P Q {s | batchScore φ reg hreg s ≤ η} ≤ ENNReal.ofReal α := by
  have hm := mean_concentration n P Q hφ hK hn hdiag hkernel
    (by linarith : 0 < α / 2) (by linarith : α / 2 < 1)
  have ho := operator_concentration n P Q hφ hK hn hdiag
    (by linarith : 0 < α / 2) (by linarith : α / 2 < 1)
  have hop : batchLaw n P Q {s | operatorRadius K n (α / 2) <
      ‖(empiricalOperator φ reg hreg s).toContinuousLinearMap -
        (populationOperator P hφ hdiag reg hreg).toContinuousLinearMap‖} ≤
      ENNReal.ofReal (α / 2) := by simpa only [operator_difference P hφ hdiag hreg] using ho
  exact alternative_error_of_concentration (batchLaw n P Q)
    (populationOperator P hφ hdiag reg hreg) (empiricalOperator φ reg hreg)
    (Moments.mean Q φ - Moments.mean P φ) (empiricalDelta φ)
    (population_positive P hφ hdiag hreg) (empirical_positive φ hreg)
    hδ hK.le hreg hα.le hseparation (population_coercive P hφ hdiag hreg)
    hm hop hthreshold

end Statistical
end PearsonClassification
