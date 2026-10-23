module

public import Mathlib
public import BoundsExtendedOperators
public import BoundsExtendedMoments

/-!
Deterministic scalar bounds and their specializations to actual operators.

This file verifies:
* the deterministic null and non-merging score bounds;
* the exact condition for a nonempty static-threshold interval;
* the union-bound argument, without independence between estimators.

The original abstract-map lemmas retain their metric hypotheses for reuse.
The quadratic-score lemmas below instead use actual positive operators and
the inverse comparison proved in `BoundsExtendedOperators`.
Concentration from the sampling assumptions is proved in
`BoundsExtendedStatistical`; see `COVERAGE.md` for the manuscript correspondence.

There are no `sorry`s or additional axioms.
-/

@[expose] public section

noncomputable section

namespace PearsonClassification

/-! ## Elementary square inequalities -/

lemma square_mono
    {a b : ℝ}
    (ha : 0 ≤ a)
    (hab : a ≤ b) :
    a ^ 2 ≤ b ^ 2 := by
  have hb : 0 ≤ b := le_trans ha hab
  have hp :
      0 ≤ (b - a) * (b + a) :=
    mul_nonneg (sub_nonneg.mpr hab) (add_nonneg hb ha)
  nlinarith

lemma square_strict_mono
    {a b : ℝ}
    (ha : 0 ≤ a)
    (hab : a < b) :
    a ^ 2 < b ^ 2 := by
  have hsum : 0 < b + a := by
    linarith
  have hp :
      0 < (b - a) * (b + a) :=
    mul_pos (sub_pos.mpr hab) hsum
  nlinarith

/-! ## Confidence radii and score bounds -/

def meanRadius (K : ℝ) (n : ℕ) (α : ℝ) : ℝ :=
  Real.sqrt (2 * K / (n : ℝ)) *
    (1 + Real.sqrt (Real.log (1 / α)))

def operatorRadius (K : ℝ) (n : ℕ) (α : ℝ) : ℝ :=
  K / Real.sqrt (n : ℝ) *
    (1 + Real.sqrt (Real.log (1 / α)))

def upperBound (b₀ reg : ℝ) : ℝ :=
  b₀ ^ 2 / (2 * reg)

def lowerBound (δ b₁ c reg : ℝ) : ℝ :=
  (max (Real.sqrt (2 * δ) - b₁ / Real.sqrt reg) 0) ^ 2 /
    (2 * c)

lemma meanRadius_pos
    {K α : ℝ} {n : ℕ}
    (hK : 0 < K)
    (hn : 0 < n) :
    0 < meanRadius K n α := by
  have hnR : 0 < (n : ℝ) := by
    exact_mod_cast hn
  unfold meanRadius
  positivity

lemma operatorRadius_nonneg
    {K α : ℝ} {n : ℕ}
    (hK : 0 ≤ K) :
    0 ≤ operatorRadius K n α := by
  unfold operatorRadius
  positivity

lemma operatorFactor_pos
    {e reg : ℝ}
    (he : 0 ≤ e)
    (hreg : 0 < reg) :
    0 < 1 + e / reg := by
  have hratio : 0 ≤ e / reg :=
    div_nonneg he (le_of_lt hreg)
  linarith

/-! ## Deterministic lower-bound calculation -/

theorem lower_bound_of_scalars
    {δ b c reg x y : ℝ}
    (hc : 0 < c)
    (hx : 0 ≤ x)
    (hmargin :
      Real.sqrt (2 * δ) ≤ x + b / Real.sqrt reg)
    (hmetric : x ^ 2 ≤ c * y ^ 2) :
    lowerBound δ b c reg ≤ y ^ 2 / 2 := by
  have hresidual :
      Real.sqrt (2 * δ) - b / Real.sqrt reg ≤ x := by
    linarith
  have hmax :
      max (Real.sqrt (2 * δ) - b / Real.sqrt reg) 0 ≤ x :=
    max_le hresidual hx
  have hmax_nonneg :
      0 ≤ max (Real.sqrt (2 * δ) - b / Real.sqrt reg) 0 :=
    le_max_right _ _
  have hsq :
      (max (Real.sqrt (2 * δ) - b / Real.sqrt reg) 0) ^ 2
        ≤ c * y ^ 2 :=
    le_trans (square_mono hmax_nonneg hmax) hmetric
  have hden : 0 < 2 * c := by
    positivity
  unfold lowerBound
  apply (div_le_iff₀ hden).2
  nlinarith [hsq]

/-!
`W` represents the population inverse square root.
`W_hat` represents the empirical inverse square root.

Thus:
  score W Δ   = (1/2) ‖A⁻¹ᐟ² Δ‖²
  score W_hat Δ_hat = (1/2) ‖Â⁻¹ᐟ² Δ_hat‖²

The metric-comparison hypotheses below encode the corresponding
operator inequalities.
-/

section Geometry

variable {E : Type*}
variable [NormedAddCommGroup E] [NormedSpace ℝ E]

def score (W : E →ₗ[ℝ] E) (v : E) : ℝ :=
  ‖W v‖ ^ 2 / 2

lemma score_nonneg
    (W : E →ₗ[ℝ] E)
    (v : E) :
    0 ≤ score W v := by
  unfold score
  positivity

theorem null_upper_bound
    (W_hat : E →ₗ[ℝ] E)
    (Δ Δ_hat : E)
    {b reg : ℝ}
    (hreg : 0 < reg)
    (hnull : Δ = 0)
    (herror : ‖Δ_hat - Δ‖ ≤ b)
    (hmetric : ∀ v : E, ‖W_hat v‖ ^ 2 ≤ ‖v‖ ^ 2 / reg) :
    score W_hat Δ_hat ≤ upperBound b reg := by
  have hnorm : ‖Δ_hat‖ ≤ b := by
    simpa [hnull] using herror
  have hsquare : ‖Δ_hat‖ ^ 2 ≤ b ^ 2 :=
    square_mono (norm_nonneg _) hnorm
  have hden : 0 ≤ 2 * reg := by
    positivity
  unfold score upperBound
  calc
    ‖W_hat Δ_hat‖ ^ 2 / 2
        ≤ (‖Δ_hat‖ ^ 2 / reg) / 2 :=
      div_le_div_of_nonneg_right (hmetric Δ_hat) (by norm_num)
    _ = ‖Δ_hat‖ ^ 2 / (2 * reg) := by
      rw [div_div, mul_comm reg 2]
    _ ≤ b ^ 2 / (2 * reg) :=
      div_le_div_of_nonneg_right hsquare hden

theorem nonmerging_lower_bound
    (W W_hat : E →ₗ[ℝ] E)
    (Δ Δ_hat : E)
    {δ b c reg : ℝ}
    (hδ : 0 ≤ δ)
    (_hreg : 0 < reg)
    (hc : 0 < c)
    (hseparation : δ ≤ score W Δ)
    (herror : ‖Δ_hat - Δ‖ ≤ b)
    (hpopulation_metric :
      ∀ v : E, ‖W v‖ ≤ ‖v‖ / Real.sqrt reg)
    (hempirical_comparison :
      ∀ v : E, ‖W v‖ ^ 2 ≤ c * ‖W_hat v‖ ^ 2) :
    lowerBound δ b c reg ≤ score W_hat Δ_hat := by
  have hroot :
      (Real.sqrt (2 * δ)) ^ 2 = 2 * δ :=
    Real.sq_sqrt (by positivity)
  have hsignal :
      Real.sqrt (2 * δ) ≤ ‖W Δ‖ := by
    by_contra h
    have hlt : ‖W Δ‖ < Real.sqrt (2 * δ) :=
      lt_of_not_ge h
    have hsq :=
      square_strict_mono (norm_nonneg (W Δ)) hlt
    unfold score at hseparation
    nlinarith [hroot, hsq]
  have hweighted_error :
      ‖W (Δ_hat - Δ)‖ ≤ b / Real.sqrt reg := by
    calc
      ‖W (Δ_hat - Δ)‖
          ≤ ‖Δ_hat - Δ‖ / Real.sqrt reg :=
        hpopulation_metric _
      _ ≤ b / Real.sqrt reg :=
        div_le_div_of_nonneg_right herror (Real.sqrt_nonneg _)
  have hidentity :
      W Δ = W Δ_hat - W (Δ_hat - Δ) := by
    rw [map_sub]
    abel
  have htriangle :
      ‖W Δ‖ ≤ ‖W Δ_hat‖ + ‖W (Δ_hat - Δ)‖ := by
    calc
      ‖W Δ‖ = ‖W Δ_hat - W (Δ_hat - Δ)‖ :=
        congrArg norm hidentity
      _ ≤ ‖W Δ_hat‖ + ‖W (Δ_hat - Δ)‖ :=
        norm_sub_le _ _
  have hmargin :
      Real.sqrt (2 * δ) ≤
        ‖W Δ_hat‖ + b / Real.sqrt reg := by
    exact le_trans hsignal
      (le_trans htriangle
        (add_le_add le_rfl hweighted_error))
  unfold score
  exact lower_bound_of_scalars hc (norm_nonneg _) hmargin
    (hempirical_comparison Δ_hat)

end Geometry

/-! ## Exact nonemptiness criterion -/

lemma scaled_square
    (x : ℝ)
    {reg : ℝ}
    (hreg : 0 < reg) :
    (x / Real.sqrt reg) ^ 2 / 2 =
      x ^ 2 / (2 * reg) := by
  rw [div_pow, Real.sq_sqrt (le_of_lt hreg)]
  rw [div_div, mul_comm reg 2]

lemma scalar_threshold_gap_iff
    {s b₀ b₁ c : ℝ}
    (hb₀ : 0 < b₀)
    (hc : 0 < c) :
    b₀ ^ 2 / 2 <
        (max (s - b₁) 0) ^ 2 / (2 * c) ↔
      b₁ + Real.sqrt c * b₀ < s := by
  have hden : 0 < 2 * c := by
    positivity
  have hk : 0 < Real.sqrt c * b₀ :=
    mul_pos (Real.sqrt_pos.2 hc) hb₀
  have hk_sq :
      (Real.sqrt c * b₀) ^ 2 = c * b₀ ^ 2 := by
    rw [mul_pow, Real.sq_sqrt (le_of_lt hc)]
  constructor
  · intro hgap
    have hsq :
        c * b₀ ^ 2 < (max (s - b₁) 0) ^ 2 := by
      have h := (lt_div_iff₀ hden).1 hgap
      nlinarith
    have hkmax :
        Real.sqrt c * b₀ < max (s - b₁) 0 := by
      by_contra h
      have hreverse :
          max (s - b₁) 0 ≤ Real.sqrt c * b₀ :=
        le_of_not_gt h
      have hreverse_sq :=
        square_mono (le_max_right (s - b₁) 0) hreverse
      nlinarith [hk_sq, hsq]
    rcases le_total (s - b₁) 0 with hnonpos | hnonneg
    · rw [max_eq_right hnonpos] at hkmax
      linarith
    · rw [max_eq_left hnonneg] at hkmax
      linarith
  · intro hmargin
    have hnonneg : 0 ≤ s - b₁ := by
      linarith
    have hkres :
        Real.sqrt c * b₀ < s - b₁ := by
      linarith
    have hsq :=
      square_strict_mono (le_of_lt hk) hkres
    rw [max_eq_left hnonneg]
    apply (lt_div_iff₀ hden).2
    nlinarith [hk_sq, hsq]

lemma sqrt_margin_iff
    {A δ reg : ℝ}
    (hA : 0 ≤ A)
    (hδ : 0 ≤ δ)
    (hreg : 0 < reg) :
    A / Real.sqrt reg < Real.sqrt (2 * δ) ↔
      A ^ 2 / (2 * reg) < δ := by
  have hsquare_div :
      (A / Real.sqrt reg) ^ 2 = A ^ 2 / reg := by
    rw [div_pow, Real.sq_sqrt (le_of_lt hreg)]
  have hsquare_root :
      (Real.sqrt (2 * δ)) ^ 2 = 2 * δ :=
    Real.sq_sqrt (by positivity)
  have hden : 0 < 2 * reg := by
    positivity
  constructor
  · intro hmargin
    have hsquare :=
      square_strict_mono
        (div_nonneg hA (Real.sqrt_nonneg reg)) hmargin
    rw [hsquare_div, hsquare_root] at hsquare
    have hmul :=
      (div_lt_iff₀ hreg).1 hsquare
    apply (div_lt_iff₀ hden).2
    nlinarith [hmul]
  · intro hgap
    have hmul :=
      (div_lt_iff₀ hden).1 hgap
    have hsquare : A ^ 2 / reg < 2 * δ := by
      apply (div_lt_iff₀ hreg).2
      nlinarith [hmul]
    by_contra h
    have hreverse :
        Real.sqrt (2 * δ) ≤ A / Real.sqrt reg :=
      le_of_not_gt h
    have hreverse_sq :=
      square_mono (Real.sqrt_nonneg (2 * δ)) hreverse
    rw [hsquare_root, hsquare_div] at hreverse_sq
    linarith

theorem threshold_gap_iff
    {δ b₀ b₁ c reg : ℝ}
    (hδ : 0 ≤ δ)
    (hb₀ : 0 < b₀)
    (hb₁ : 0 ≤ b₁)
    (hc : 0 < c)
    (hreg : 0 < reg) :
    upperBound b₀ reg < lowerBound δ b₁ c reg ↔
      (b₁ + Real.sqrt c * b₀) ^ 2 / (2 * reg) < δ := by
  have hscaled_b₀ :
      0 < b₀ / Real.sqrt reg :=
    div_pos hb₀ (Real.sqrt_pos.2 hreg)
  have hA :
      0 ≤ b₁ + Real.sqrt c * b₀ := by
    positivity
  have hbase :=
    scalar_threshold_gap_iff
      (s := Real.sqrt (2 * δ))
      (b₀ := b₀ / Real.sqrt reg)
      (b₁ := b₁ / Real.sqrt reg)
      (c := c)
      hscaled_b₀ hc
  rw [scaled_square b₀ hreg] at hbase
  have hidentity :
      b₁ / Real.sqrt reg +
          Real.sqrt c * (b₀ / Real.sqrt reg) =
        (b₁ + Real.sqrt c * b₀) / Real.sqrt reg := by
    ring
  rw [hidentity] at hbase
  change
    b₀ ^ 2 / (2 * reg) <
        (max (Real.sqrt (2 * δ) - b₁ / Real.sqrt reg) 0) ^ 2 /
          (2 * c) ↔
      (b₁ + Real.sqrt c * b₀) ^ 2 / (2 * reg) < δ
  exact hbase.trans (sqrt_margin_iff hA hδ hreg)

theorem exists_static_threshold_iff
    {δ b₀ b₁ c reg : ℝ}
    (hδ : 0 ≤ δ)
    (hb₀ : 0 < b₀)
    (hb₁ : 0 ≤ b₁)
    (hc : 0 < c)
    (hreg : 0 < reg) :
    (∃ η : ℝ,
      upperBound b₀ reg ≤ η ∧
      η < lowerBound δ b₁ c reg) ↔
      (b₁ + Real.sqrt c * b₀) ^ 2 / (2 * reg) < δ := by
  constructor
  · rintro ⟨η, hupper, hlower⟩
    exact
      (threshold_gap_iff hδ hb₀ hb₁ hc hreg).1
        (lt_of_le_of_lt hupper hlower)
  · intro hseparation
    have hgap :=
      (threshold_gap_iff hδ hb₀ hb₁ hc hreg).2 hseparation
    exact ⟨upperBound b₀ reg, le_rfl, hgap⟩

/-! ## Probability bounds without independence between estimators -/

section Probability

open MeasureTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω]

/-!
These results use outer-measure inequalities, so measurability of the
individual events is not needed for the union-bound calculation.
For probabilistic interpretation, take `μ` to be a probability measure.
-/

theorem false_separation_measure_bound
    (μ : Measure Ω)
    (S : Ω → ℝ)
    (goodMean : Set Ω)
    {U η : ℝ}
    (hthreshold : U ≤ η)
    (hgood : ∀ ω ∈ goodMean, S ω ≤ U) :
    μ {ω | η < S ω} ≤ μ goodMeanᶜ := by
  apply measure_mono
  intro ω hfailure
  change ω ∉ goodMean
  intro hmean
  have hscore := hgood ω hmean
  exact (not_lt_of_ge (le_trans hscore hthreshold)) hfailure

theorem false_merging_measure_bound
    (μ : Measure Ω)
    (S : Ω → ℝ)
    (goodMean goodOperator : Set Ω)
    {L η : ℝ}
    (hthreshold : η < L)
    (hgood :
      ∀ ω, ω ∈ goodMean → ω ∈ goodOperator → L ≤ S ω) :
    μ {ω | S ω ≤ η} ≤
      μ goodMeanᶜ + μ goodOperatorᶜ := by
  have hsubset :
      {ω | S ω ≤ η} ⊆ goodMeanᶜ ∪ goodOperatorᶜ := by
    intro ω hfailure
    by_cases hmean : ω ∈ goodMean
    · right
      change ω ∉ goodOperator
      intro hop
      have hlower := hgood ω hmean hop
      exact
        (not_le_of_gt hthreshold)
          (le_trans hlower hfailure)
    · left
      exact hmean
  calc
    μ {ω | S ω ≤ η}
        ≤ μ (goodMeanᶜ ∪ goodOperatorᶜ) :=
      measure_mono hsubset
    _ ≤ μ goodMeanᶜ + μ goodOperatorᶜ :=
      measure_union_le _ _

theorem false_merging_confidence_bound
    (μ : Measure Ω)
    (S : Ω → ℝ)
    (goodMean goodOperator : Set Ω)
    {L η : ℝ}
    {αmean αoperator : ℝ≥0∞}
    (hthreshold : η < L)
    (hgood :
      ∀ ω, ω ∈ goodMean → ω ∈ goodOperator → L ≤ S ω)
    (hmean_probability : μ goodMeanᶜ ≤ αmean)
    (hoperator_probability : μ goodOperatorᶜ ≤ αoperator) :
    μ {ω | S ω ≤ η} ≤ αmean + αoperator := by
  exact
    le_trans
      (false_merging_measure_bound
        μ S goodMean goodOperator hthreshold hgood)
      (add_le_add hmean_probability hoperator_probability)

end Probability

/-! ## Bounds for actual positive operators -/

section QuadraticGeometry

open Operators
open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem null_quadratic_upper_bound
    (B : E ≃L[ℝ] E) (Δ Δ_hat : E) {b reg : ℝ}
    (hreg : 0 < reg) (hnull : Δ = 0)
    (herror : ‖Δ_hat - Δ‖ ≤ b)
    (hcoercive : ∀ f, reg * ‖f‖ ^ 2 ≤ ⟪f, B f⟫) :
    quadraticScore B Δ_hat ≤ upperBound b reg := by
  have hn : ‖Δ_hat‖ ≤ b := by simpa [hnull] using herror
  exact (quadraticScore_upper B hreg hcoercive Δ_hat).trans
    (div_le_div_of_nonneg_right (square_mono (norm_nonneg _) hn) (by positivity))

theorem nonmerging_quadratic_lower_bound
    (A B : E ≃L[ℝ] E) (Δ Δ_hat : E) {δ b e reg : ℝ}
    (hA : A.toContinuousLinearMap.IsPositive)
    (hB : B.toContinuousLinearMap.IsPositive)
    (_hδ : 0 ≤ δ) (hreg : 0 < reg) (he : 0 ≤ e)
    (hseparation : δ ≤ quadraticScore A Δ)
    (herror : ‖Δ_hat - Δ‖ ≤ b)
    (hcoercive : ∀ f, reg * ‖f‖ ^ 2 ≤ ⟪f, A f⟫)
    (hoperator : ‖B.toContinuousLinearMap - A.toContinuousLinearMap‖ ≤ e) :
    lowerBound δ b (1 + e / reg) reg ≤ quadraticScore B Δ_hat := by
  have hc := operatorFactor_pos he hreg
  have hsignal : Real.sqrt (2 * δ) ≤ weightedNorm A Δ := by
    rw [Real.sqrt_le_iff]
    exact ⟨weightedNorm_nonneg A Δ, by rw [weightedNorm_sq A hA]; linarith⟩
  have htriangle := weightedNorm_triangle A hA Δ_hat (Δ - Δ_hat)
  have hid : Δ_hat + (Δ - Δ_hat) = Δ := by abel
  rw [hid] at htriangle
  have herr : weightedNorm A (Δ - Δ_hat) ≤ b / Real.sqrt reg := by
    apply (weightedNorm_upper A hA hreg hcoercive _).trans
    apply div_le_div_of_nonneg_right _ (Real.sqrt_nonneg _)
    simpa only [norm_sub_rev] using herror
  have hmargin : Real.sqrt (2 * δ) ≤ weightedNorm A Δ_hat + b / Real.sqrt reg := by
    linarith
  have hcomparison := inverse_comparison A B hA hB hc
    (perturbation_order A B hreg he hcoercive hoperator) Δ_hat
  have hmetric : weightedNorm A Δ_hat ^ 2 ≤
      (1 + e / reg) * weightedNorm B Δ_hat ^ 2 := by
    rw [weightedNorm_sq A hA, weightedNorm_sq B hB]
    nlinarith
  have hlower := lower_bound_of_scalars hc (weightedNorm_nonneg A Δ_hat) hmargin hmetric
  rwa [weightedNorm_sq B hB, mul_div_cancel_left₀ _ (by norm_num : (2 : ℝ) ≠ 0)] at hlower

end QuadraticGeometry

/-! ## The exact radii used in the manuscript -/

def nullBound (K : ℝ) (n : ℕ) (reg α : ℝ) : ℝ :=
  upperBound (meanRadius K n α) reg

def alternativeBound (δ K : ℝ) (n : ℕ) (reg α : ℝ) : ℝ :=
  lowerBound δ (meanRadius K n (α / 2))
    (1 + operatorRadius K n (α / 2) / reg) reg

theorem calibrated_threshold_iff {δ K reg α₀ α₁ : ℝ} {n : ℕ}
    (hδ : 0 ≤ δ) (hK : 0 < K) (hn : 0 < n) (hreg : 0 < reg) :
    (∃ η : ℝ, nullBound K n reg α₀ ≤ η ∧ η < alternativeBound δ K n reg α₁) ↔
    (meanRadius K n (α₁ / 2) +
      Real.sqrt (1 + operatorRadius K n (α₁ / 2) / reg) * meanRadius K n α₀) ^ 2 /
        (2 * reg) < δ := by
  apply exists_static_threshold_iff hδ (meanRadius_pos hK hn)
    (le_of_lt (meanRadius_pos hK hn))
    (operatorFactor_pos (operatorRadius_nonneg hK.le) hreg) hreg

end PearsonClassification
