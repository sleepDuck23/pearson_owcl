module

public import BoundsExtendedOperators

/-!
Feature-map and Bochner-moment foundations. A measurable bounded feature map
is used directly, so these statements also apply to the canonical RKHS map.
No expectation, positivity, or invertibility assertion is added as an axiom.
-/

@[expose] public section

noncomputable section

namespace PearsonClassification
namespace Moments

open MeasureTheory InnerProductSpace
open scoped RealInnerProductSpace

variable {X E : Type*} [MeasurableSpace X]
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

def mean (μ : Measure X) (φ : X → E) : E := ∫ x, φ x ∂μ

def secondMoment (μ : Measure X) (φ : X → E) : E →L[ℝ] E :=
  ∫ x, rankOne ℝ (φ x) (φ x) ∂μ

def empiricalMean {n : ℕ} (φ : X → E) (sample : Fin n → X) : E :=
  (n : ℝ)⁻¹ • ∑ i, φ (sample i)

def empiricalSecondMoment {n : ℕ} (φ : X → E) (sample : Fin n → X) : E →L[ℝ] E :=
  (n : ℝ)⁻¹ • ∑ i, rankOne ℝ (φ (sample i)) (φ (sample i))

def rkhsFeature [RKHS ℝ E X ℝ] (x : X) : E := RKHS.kerFun E x 1

omit [MeasurableSpace X] in
theorem rkhs_reproducing [RKHS ℝ E X ℝ] (x : X) (f : E) :
    ⟪f, rkhsFeature x⟫ = f x := by
  simp [rkhsFeature]

omit [MeasurableSpace X] [InnerProductSpace ℝ E] [CompleteSpace E] in
lemma feature_norm_bound {φ : X → E} {K : ℝ}
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (x : X) : ‖φ x‖ ≤ Real.sqrt K :=
  (Real.le_sqrt (norm_nonneg _) (le_trans (sq_nonneg _) (hdiag x))).2 (hdiag x)

omit [InnerProductSpace ℝ E] [CompleteSpace E] in
theorem feature_integrable (μ : Measure X) [IsProbabilityMeasure μ]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) : Integrable φ μ :=
  Integrable.of_bound hφ.aestronglyMeasurable (Real.sqrt K)
    (ae_of_all μ (feature_norm_bound hdiag))

omit [CompleteSpace E] in
theorem moment_integrable (μ : Measure X) [IsProbabilityMeasure μ]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) :
    Integrable (fun x => rankOne ℝ (φ x) (φ x)) μ := by
  refine Integrable.of_bound ?_ K ?_
  · have hc : Continuous (fun u : E => rankOne ℝ u u) :=
      (rankOne ℝ).continuous.clm_apply continuous_id
    exact (hc.comp_stronglyMeasurable hφ).aestronglyMeasurable
  · exact ae_of_all μ fun x => by simpa [norm_rankOne, pow_two] using hdiag x

theorem mean_inner (μ : Measure X) [IsProbabilityMeasure μ]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (f : E) :
    ⟪f, mean μ φ⟫ = ∫ x, ⟪f, φ x⟫ ∂μ :=
  (integral_inner (feature_integrable μ hφ hdiag) f).symm

theorem secondMoment_inner (μ : Measure X) [IsProbabilityMeasure μ]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (f g : E) :
    ⟪f, secondMoment μ φ g⟫ = ∫ x, ⟪f, φ x⟫ * ⟪φ x, g⟫ ∂μ := by
  have hi := moment_integrable μ hφ hdiag
  have hgi := (ContinuousLinearMap.apply ℝ E g).integrable_comp hi
  change Integrable (fun x => rankOne ℝ (φ x) (φ x) g) μ at hgi
  rw [secondMoment, ContinuousLinearMap.integral_apply hi,
    ← integral_inner hgi f]
  simp only [rankOne_apply, inner_smul_right]
  congr 1
  funext x
  ring

theorem secondMoment_positive (μ : Measure X) [IsProbabilityMeasure μ]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) : (secondMoment μ φ).IsPositive := by
  apply (ContinuousLinearMap.isPositive_iff _).2
  constructor
  · intro f g
    change ⟪secondMoment μ φ f, g⟫ = ⟪f, secondMoment μ φ g⟫
    rw [real_inner_comm g (secondMoment μ φ f), secondMoment_inner μ hφ hdiag,
      secondMoment_inner μ hφ hdiag]
    congr 1
    funext x
    rw [real_inner_comm g (φ x), real_inner_comm f (φ x)]
    ring
  · intro f
    rw [real_inner_comm f (secondMoment μ φ f), secondMoment_inner μ hφ hdiag]
    apply integral_nonneg
    intro x
    change 0 ≤ ⟪f, φ x⟫ * ⟪φ x, f⟫
    rw [real_inner_comm f (φ x)]
    exact mul_self_nonneg _

theorem secondMoment_quadratic (μ : Measure X) [IsProbabilityMeasure μ]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (f : E) :
    ⟪f, secondMoment μ φ f⟫ = ∫ x, ⟪f, φ x⟫ ^ 2 ∂μ := by
  rw [secondMoment_inner μ hφ hdiag]
  congr 1
  funext x
  rw [real_inner_comm f (φ x)]
  ring

theorem secondMoment_upper (μ : Measure X) [IsProbabilityMeasure μ]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (f : E) :
    ⟪f, secondMoment μ φ f⟫ ≤ K * ‖f‖ ^ 2 := by
  rw [secondMoment_quadratic μ hφ hdiag]
  calc
    (∫ x, ⟪f, φ x⟫ ^ 2 ∂μ) ≤ ∫ _ : X, K * ‖f‖ ^ 2 ∂μ := by
      apply integral_mono_of_nonneg (ae_of_all μ fun x => sq_nonneg _) (integrable_const _)
      apply ae_of_all μ
      intro x
      have hcs := abs_real_inner_le_norm f (φ x)
      have hs := sq_le_sq₀ (abs_nonneg ⟪f, φ x⟫) (by positivity) |>.2 hcs
      have hd := mul_le_mul_of_nonneg_left (hdiag x) (sq_nonneg ‖f‖)
      nlinarith [sq_abs ⟪f, φ x⟫]
    _ = K * ‖f‖ ^ 2 := by simp

omit [MeasurableSpace X] [CompleteSpace E] in
theorem empiricalSecondMoment_positive {n : ℕ} (φ : X → E) (sample : Fin n → X) :
    (empiricalSecondMoment φ sample).IsPositive := by
  apply ContinuousLinearMap.IsPositive.smul_of_nonneg _ (by positivity)
  exact ContinuousLinearMap.isPositive_sum _ fun i _ => isPositive_rankOne_self _

omit [MeasurableSpace X] [CompleteSpace E] in
/-- The improved feature diameter uses pointwise kernel nonnegativity. -/
theorem feature_diameter {φ : X → E} {K : ℝ}
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K)
    (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫) (x y : X) :
    ‖φ x - φ y‖ ^ 2 ≤ 2 * K := by
  rw [norm_sub_sq_real]
  linarith [hdiag x, hdiag y, hkernel x y]

omit [CompleteSpace E] in
/-- The score can vanish for distinct distributions if their means agree. -/
theorem score_zero_of_equal_means (A : E ≃L[ℝ] E) (P Q : Measure X) (φ : X → E)
    (heq : mean Q φ = mean P φ) :
    Operators.quadraticScore A (mean Q φ - mean P φ) = 0 := by
  simp [heq, Operators.quadraticScore]

omit [InnerProductSpace ℝ E] [CompleteSpace E] in
theorem normSquare_integrable (μ : Measure X) [IsProbabilityMeasure μ]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) : Integrable (fun x => ‖φ x‖ ^ 2) μ := by
  have hc : Continuous (fun u : E => ‖u‖ ^ 2) := by fun_prop
  refine Integrable.of_bound (hc.comp_stronglyMeasurable hφ).aestronglyMeasurable K ?_
  exact ae_of_all μ fun x => by
    change |‖φ x‖ ^ 2| ≤ K
    rw [abs_of_nonneg (sq_nonneg ‖φ x‖)]
    exact hdiag x

/-- Centering is an expectation identity, not an assumption that the mean is zero. -/
theorem centered_integral (μ : Measure X) [IsProbabilityMeasure μ]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) : ∫ x, φ x - mean μ φ ∂μ = 0 := by
  rw [integral_sub (feature_integrable μ hφ hdiag) (integrable_const _)]
  simp [mean]

theorem feature_variance_identity (μ : Measure X) [IsProbabilityMeasure μ]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) :
    (∫ x, ‖φ x - mean μ φ‖ ^ 2 ∂μ) =
      (∫ x, ‖φ x‖ ^ 2 ∂μ) - ‖mean μ φ‖ ^ 2 := by
  have hi := feature_integrable μ hφ hdiag
  have hsq := normSquare_integrable μ hφ hdiag
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

theorem feature_variance_upper (μ : Measure X) [IsProbabilityMeasure μ]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) :
    (∫ x, ‖φ x - mean μ φ‖ ^ 2 ∂μ) ≤ K := by
  rw [feature_variance_identity μ hφ hdiag]
  have hbound : (∫ x, ‖φ x‖ ^ 2 ∂μ) ≤ K := by
    calc
      (∫ x, ‖φ x‖ ^ 2 ∂μ) ≤ ∫ _ : X, K ∂μ :=
        integral_mono (normSquare_integrable μ hφ hdiag) (integrable_const _) hdiag
      _ = K := by simp
  linarith [sq_nonneg ‖mean μ φ‖]

/-- Appendix `eq:score-MMD`: both sides for the actual regularized moment operator. -/
theorem score_MMD_bounds (μ : Measure X) [IsProbabilityMeasure μ]
    {φ : X → E} {K reg : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hK : 0 ≤ K) (hreg : 0 < reg) (v : E) :
    let A := Operators.regularizedEquiv (secondMoment μ φ)
      (secondMoment_positive μ hφ hdiag) reg hreg
    ‖v‖ ^ 2 / (2 * (K + reg)) ≤ Operators.quadraticScore A v ∧
      Operators.quadraticScore A v ≤ ‖v‖ ^ 2 / (2 * reg) := by
  dsimp only
  let A := Operators.regularizedEquiv (secondMoment μ φ)
    (secondMoment_positive μ hφ hdiag) reg hreg
  have hA : A.toContinuousLinearMap = Operators.regularized (secondMoment μ φ) reg :=
    Operators.regularizedEquiv_coe _ _ _ _
  have hpos : A.toContinuousLinearMap.IsPositive := by
    rw [hA]
    exact Operators.regularized_positive _ (secondMoment_positive μ hφ hdiag) hreg.le
  have hcoercive : ∀ f, reg * ‖f‖ ^ 2 ≤ ⟪f, A f⟫ := by
    intro f
    change reg * ‖f‖ ^ 2 ≤ ⟪f, A.toContinuousLinearMap f⟫
    rw [hA]
    exact Operators.regularized_coercive _ (secondMoment_positive μ hφ hdiag) reg f
  constructor
  · apply Operators.quadraticScore_lower A hpos (by positivity : 0 < K + reg)
    intro f
    change ⟪f, A.toContinuousLinearMap f⟫ ≤ _
    rw [hA]
    simp only [Operators.regularized, add_apply, smul_apply, ContinuousLinearMap.id_apply,
      inner_add_right, inner_smul_right, real_inner_self_eq_norm_sq]
    nlinarith [secondMoment_upper μ hφ hdiag f]
  · exact Operators.quadraticScore_upper A hreg hcoercive v

section Independence

variable [MeasurableSpace E] [BorelSpace E]

theorem independent_centered_inner (μ : Measure X) [IsProbabilityMeasure μ]
    {U V : X → E} (hUV : ProbabilityTheory.IndepFun U V μ)
    (hU : Integrable U μ) (hV : Integrable V μ)
    (hcenterU : ∫ x, U x ∂μ = 0) (hcenterV : ∫ x, V x ∂μ = 0) :
    ∫ x, ⟪U x, V x⟫ ∂μ = 0 := by
  have h := hUV.integral_bilin hU hV (innerSL ℝ)
  change (∫ x, ⟪U x, V x⟫ ∂μ) = ⟪∫ x, U x ∂μ, ∫ x, V x ∂μ⟫ at h
  simpa only [hcenterU, hcenterV, inner_zero_left] using h

end Independence

def mixture (P R : Measure X) (ε : ℝ) : Measure X :=
  ENNReal.ofReal (1 - ε) • P + ENNReal.ofReal ε • R

omit [CompleteSpace E] in
theorem mean_mixture (P R : Measure X) [IsProbabilityMeasure P] [IsProbabilityMeasure R]
    {φ : X → E} {K ε : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hε : 0 ≤ ε) (hε₁ : ε ≤ 1) :
    mean (mixture P R ε) φ = (1 - ε) • mean P φ + ε • mean R φ := by
  unfold mean mixture
  rw [integral_add_measure ((feature_integrable P hφ hdiag).smul_measure ENNReal.ofReal_ne_top)
    ((feature_integrable R hφ hdiag).smul_measure ENNReal.ofReal_ne_top),
    integral_smul_measure, integral_smul_measure,
    ENNReal.toReal_ofReal (by linarith : 0 ≤ 1 - ε), ENNReal.toReal_ofReal hε]

omit [CompleteSpace E] in
theorem mixture_score_scaling (P R : Measure X) [IsProbabilityMeasure P] [IsProbabilityMeasure R]
    (A : E ≃L[ℝ] E) {φ : X → E} {K ε : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hε : 0 ≤ ε) (hε₁ : ε ≤ 1) :
    Operators.quadraticScore A (mean (mixture P R ε) φ - mean P φ) =
      ε ^ 2 * Operators.quadraticScore A (mean R φ - mean P φ) := by
  rw [mean_mixture P R hφ hdiag hε hε₁]
  have hid : (1 - ε) • mean P φ + ε • mean R φ - mean P φ =
      ε • (mean R φ - mean P φ) := by module
  rw [hid, Operators.quadraticScore_smul]

theorem moment_variational_objective (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (reg : ℝ) (f : E) :
    Operators.objective (Operators.regularized (secondMoment P φ) reg)
      (mean Q φ - mean P φ) f =
      (∫ x, ⟪f, φ x⟫ ∂Q) - (∫ x, ⟪f, φ x⟫ ∂P) -
        (∫ x, ⟪f, φ x⟫ ^ 2 ∂P) / 2 - reg / 2 * ‖f‖ ^ 2 := by
  simp only [Operators.objective, Operators.regularized, add_apply, smul_apply,
    ContinuousLinearMap.id_apply, inner_sub_right, inner_add_right, inner_smul_right,
    real_inner_self_eq_norm_sq, mean_inner P hφ hdiag, mean_inner Q hφ hdiag,
    secondMoment_quadratic P hφ hdiag]
  ring

def populationScore (P Q : Measure X) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {K : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (reg : ℝ) (hreg : 0 < reg) : ℝ :=
  Operators.quadraticScore
    (Operators.regularizedEquiv (secondMoment P φ) (secondMoment_positive P hφ hdiag) reg hreg)
    (mean Q φ - mean P φ)

def empiricalScore {n : ℕ} (φ : X → E) (reference incoming : Fin n → X)
    (reg : ℝ) (hreg : 0 < reg) : ℝ :=
  Operators.quadraticScore
    (Operators.regularizedEquiv (empiricalSecondMoment φ reference)
      (empiricalSecondMoment_positive φ reference) reg hreg)
    (empiricalMean φ incoming - empiricalMean φ reference)

theorem populationScore_variational (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {K reg : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hreg : 0 < reg) :
    sSup (Set.range (fun f : E =>
      (∫ x, ⟪f, φ x⟫ ∂Q) - (∫ x, ⟪f, φ x⟫ ∂P) -
        (∫ x, ⟪f, φ x⟫ ^ 2 ∂P) / 2 - reg / 2 * ‖f‖ ^ 2)) =
      populationScore P Q hφ hdiag reg hreg := by
  let A := Operators.regularizedEquiv (secondMoment P φ)
    (secondMoment_positive P hφ hdiag) reg hreg
  have hA : A.toContinuousLinearMap = Operators.regularized (secondMoment P φ) reg :=
    Operators.regularizedEquiv_coe _ _ _ _
  have hpos : A.toContinuousLinearMap.IsPositive := by
    rw [hA]
    exact Operators.regularized_positive _ (secondMoment_positive P hφ hdiag) hreg.le
  have hid : (fun f : E => (∫ x, ⟪f, φ x⟫ ∂Q) - (∫ x, ⟪f, φ x⟫ ∂P) -
      (∫ x, ⟪f, φ x⟫ ^ 2 ∂P) / 2 - reg / 2 * ‖f‖ ^ 2) =
      Operators.objective A.toContinuousLinearMap (mean Q φ - mean P φ) := by
    funext f
    rw [hA, moment_variational_objective P Q hφ hdiag]
  rw [hid]
  exact Operators.variational_supremum A hpos _

end Moments
end PearsonClassification
