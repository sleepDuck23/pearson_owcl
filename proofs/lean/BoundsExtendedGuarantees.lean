module

public import BoundsExtendedSamplingModel
public import BoundsExtendedFoundations

/-! Final classification guarantees, including finite banks of coupled scores. -/

@[expose] public section
noncomputable section
namespace PearsonClassification.Statistical
open MeasureTheory BoundedDifferences
open scoped RealInnerProductSpace ENNReal
variable {X E : Type*} [MeasurableSpace X]
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- One predetermined threshold controls both population scenarios. -/
theorem static_threshold (n : ℕ) (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    [MeasurableSpace E] [BorelSpace E] [TopologicalSpace.SeparableSpace E]
    {φ : X → E} {δ K reg α₀ α₁ η : ℝ} (hφ : Measurable φ)
    (hK : 0 < K) (hn : 0 < n) (hreg : 0 < reg) (hδ : 0 ≤ δ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫)
    (hα₀ : 0 < α₀) (hα₀₁ : α₀ < 1) (hα₁ : 0 < α₁) (hα₁₁ : α₁ < 1)
    (hseparation : δ ≤ Moments.populationScore P Q hφ.stronglyMeasurable hdiag reg hreg)
    (hlow : nullBound K n reg α₀ ≤ η) (hhigh : η < alternativeBound δ K n reg α₁) :
    batchLaw n P P {s | η < batchScore φ reg hreg s} ≤ ENNReal.ofReal α₀ ∧
    batchLaw n P Q {s | batchScore φ reg hreg s ≤ η} ≤ ENNReal.ofReal α₁ :=
  ⟨null_error n P hφ.stronglyMeasurable hK hn hreg hdiag hkernel hα₀ hα₀₁ hlow,
   alternative_error n P Q hφ.stronglyMeasurable hK hn hreg hδ hdiag hkernel
     hα₁ hα₁₁ hseparation hhigh⟩

theorem null_high_probability (n : ℕ) (P : Measure X) [IsProbabilityMeasure P]
    {φ : X → E} {K reg α : ℝ} (hφ : StronglyMeasurable φ)
    (hK : 0 < K) (hn : 0 < n) (hreg : 0 < reg)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫)
    (hα : 0 < α) (hα₁ : α < 1) :
    1 - ENNReal.ofReal α ≤ batchLaw n P P {s | 0 ≤ batchScore φ reg hreg s ∧
      batchScore φ reg hreg s ≤ nullBound K n reg α} := by
  exact null_high_probability_of_concentration (batchLaw n P P)
    (empiricalOperator φ reg hreg) (Moments.mean P φ - Moments.mean P φ) (empiricalDelta φ)
    (empirical_positive φ hreg) hreg (sub_self _) (empirical_coercive φ hreg)
    (mean_concentration n P P hφ hK hn hdiag hkernel hα hα₁)

theorem alternative_high_probability (n : ℕ) (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {δ K reg α : ℝ} (hφ : StronglyMeasurable φ)
    (hK : 0 < K) (hn : 0 < n) (hreg : 0 < reg) (hδ : 0 ≤ δ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫)
    (hα : 0 < α) (hα₁ : α < 1)
    (hseparation : δ ≤ Moments.populationScore P Q hφ hdiag reg hreg) :
    1 - ENNReal.ofReal α ≤ batchLaw n P Q {s | alternativeBound δ K n reg α ≤
      batchScore φ reg hreg s} := by
  have hm := mean_concentration n P Q hφ hK hn hdiag hkernel
    (by linarith : 0 < α / 2) (by linarith : α / 2 < 1)
  have ho := operator_concentration n P Q hφ hK hn hdiag
    (by linarith : 0 < α / 2) (by linarith : α / 2 < 1)
  have hop : batchLaw n P Q {s | operatorRadius K n (α / 2) <
      ‖(empiricalOperator φ reg hreg s).toContinuousLinearMap -
        (populationOperator P hφ hdiag reg hreg).toContinuousLinearMap‖} ≤
      ENNReal.ofReal (α / 2) := by simpa only [operator_difference P hφ hdiag hreg] using ho
  exact alternative_high_probability_of_concentration (batchLaw n P Q)
    (populationOperator P hφ hdiag reg hreg) (empiricalOperator φ reg hreg)
    (Moments.mean Q φ - Moments.mean P φ) (empiricalDelta φ)
    (population_positive P hφ hdiag hreg) (empirical_positive φ hreg)
    hδ hK.le hreg hα.le hseparation (population_coercive P hφ hdiag hreg) hm hop

section Bank
variable {Ω ι : Type*} [MeasurableSpace Ω] [Fintype ι] [Nonempty ι]

/-- Transport an error bound along a measurable sampling map. No independence
of the different comparisons is required. -/
theorem transport_event_bound (ν : Measure Ω) {S : Type*} [MeasurableSpace S]
    (sample : Ω → S) (hm : Measurable sample) (μ : Measure S)
    (hlaw : ν.map sample = μ) (event : Set S) {ε : ℝ≥0∞} (h : μ event ≤ ε) :
    ν (sample ⁻¹' event) ≤ ε := by
  have hmap := Measure.le_map_apply (μ := ν) hm.aemeasurable event
  rw [hlaw] at hmap
  exact hmap.trans h

/-- The fixed-bank theorem from independent observations and their marginals,
with actual empirical Pearson scores and the manuscript's `α/M` allocation. -/
theorem bank_unknown (ν : Measure Ω) [IsProbabilityMeasure ν] (n : ℕ)
    (P : ι → Measure X) (Q : Measure X)
    [∀ i, IsProbabilityMeasure (P i)] [IsProbabilityMeasure Q]
    (Z : ι → Fin (n + n) → Ω → X)
    (hZ : ∀ i j, Measurable (Z i j))
    (hind : ∀ i, ProbabilityTheory.iIndepFun (Z i) ν)
    (hmarginal : ∀ i j, ν.map (Z i j) = batchMeasures n (P i) Q j)
    {φ : X → E} {δ K reg α η : ℝ} (hφ : StronglyMeasurable φ)
    (hK : 0 < K) (hn : 0 < n) (hreg : 0 < reg) (hδ : 0 ≤ δ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫)
    (hα : 0 < α) (hα₁ : α < 1)
    (hsep : ∀ i, δ ≤ Moments.populationScore (P i) Q hφ hdiag reg hreg)
    (hthreshold : η < alternativeBound δ K n reg (α / (Fintype.card ι : ℝ))) :
    1 - ENNReal.ofReal α ≤ ν {ω | acceptedReferences
      (fun i ω => batchScore φ reg hreg (pack (n + n) (fun j => Z i j ω))) η ω = ∅} := by
  have hc : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have hc₁ : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have hε : 0 < α / (Fintype.card ι : ℝ) := div_pos hα hc
  have hε₁ : α / (Fintype.card ι : ℝ) < 1 := (div_lt_one hc).2 (hα₁.trans_le hc₁)
  apply bank_unknown_calibrated ν _ η hα.le
  intro i
  exact transport_event_bound ν _
    ((measurable_pack _).comp (measurable_pi_iff.mpr (hZ i))) (batchLaw n (P i) Q)
    (independent_samples_law ν _ _ (Z i) (hZ i) (hind i) (hmarginal i)) _
    (alternative_error n (P i) Q hφ hK hn hreg hδ hdiag hkernel hε hε₁
      (hsep i) hthreshold)

theorem bank_known (ν : Measure Ω) [IsProbabilityMeasure ν] (n : ℕ)
    (P : ι → Measure X) [∀ i, IsProbabilityMeasure (P i)] (j : ι)
    (Z : ι → Fin (n + n) → Ω → X)
    (hZ : ∀ i k, Measurable (Z i k))
    (hind : ∀ i, ProbabilityTheory.iIndepFun (Z i) ν)
    (hmarginal : ∀ i k, ν.map (Z i k) = batchMeasures n (P i) (P j) k)
    {φ : X → E} {δ K reg α η : ℝ} (hφ : StronglyMeasurable φ)
    (hK : 0 < K) (hn : 0 < n) (hreg : 0 < reg) (hδ : 0 ≤ δ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫)
    (hα : 0 < α) (hα₁ : α < 1)
    (hsep : ∀ i, i ≠ j → δ ≤ Moments.populationScore (P i) (P j) hφ hdiag reg hreg)
    (hlow : nullBound K n reg (α / (Fintype.card ι : ℝ)) ≤ η)
    (hhigh : η < alternativeBound δ K n reg (α / (Fintype.card ι : ℝ))) :
    1 - ENNReal.ofReal α ≤ ν {ω | acceptedReferences
      (fun i ω => batchScore φ reg hreg (pack (n + n) (fun k => Z i k ω))) η ω = {j}} := by
  have hc : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have hc₁ : (1 : ℝ) ≤ Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have hε : 0 < α / (Fintype.card ι : ℝ) := div_pos hα hc
  have hε₁ : α / (Fintype.card ι : ℝ) < 1 := (div_lt_one hc).2 (hα₁.trans_le hc₁)
  apply bank_known_calibrated ν _ η j hα.le
  · exact transport_event_bound ν _
      ((measurable_pack _).comp (measurable_pi_iff.mpr (hZ j))) (batchLaw n (P j) (P j))
      (independent_samples_law ν _ _ (Z j) (hZ j) (hind j) (hmarginal j)) _
      (null_error n (P j) hφ hK hn hreg hdiag hkernel hε hε₁ hlow)
  · intro i hij
    exact transport_event_bound ν _
      ((measurable_pack _).comp (measurable_pi_iff.mpr (hZ i))) (batchLaw n (P i) (P j))
      (independent_samples_law ν _ _ (Z i) (hZ i) (hind i) (hmarginal i)) _
      (alternative_error n (P i) (P j) hφ hK hn hreg hδ hdiag hkernel hε hε₁
        (hsep i hij) hhigh)

end Bank
end PearsonClassification.Statistical
