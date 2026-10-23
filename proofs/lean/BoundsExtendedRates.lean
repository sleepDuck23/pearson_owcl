module

public import BoundsExtendedCalibration
public import BoundsExtendedStatistical

/-! Rates for varying regularization and positive separation of finite families. -/

@[expose] public section
noncomputable section
namespace PearsonClassification
open MeasureTheory
open scoped RealInnerProductSpace

/-- An exact identity, stronger than the first asymptotic rate. -/
theorem meanRadius_scaled (K α reg : ℝ) (n : ℕ) (hK : 0 ≤ K) :
    meanRadius K n α / Real.sqrt reg =
      (Real.sqrt (2 * K) * confidenceMultiplier α) *
        (1 / Real.sqrt ((n : ℝ) * reg)) := by
  rw [meanRadius, Real.sqrt_div (by positivity),
    Real.sqrt_mul (Nat.cast_nonneg n)]
  unfold confidenceMultiplier
  ring

theorem operatorRadius_scaled (K α reg : ℝ) (n : ℕ) :
    operatorRadius K n α / reg = (K * confidenceMultiplier α) *
      (1 / (reg * Real.sqrt (n : ℝ))) := by
  unfold operatorRadius confidenceMultiplier
  ring

/-- The rate holds even when regularization varies with the sample size. -/
theorem meanRadius_isBigO {K : ℝ} (hK : 0 ≤ K) (α : ℝ) (reg : ℕ → ℝ) :
    (fun n => meanRadius K n α / Real.sqrt (reg n)) =O[Filter.atTop]
      (fun n => 1 / Real.sqrt ((n : ℝ) * reg n)) := by
  apply Asymptotics.IsBigO.of_bound ‖Real.sqrt (2 * K) * confidenceMultiplier α‖
  exact Filter.Eventually.of_forall fun n => by rw [meanRadius_scaled _ _ _ _ hK, norm_mul]

theorem operatorRadius_isBigO (K α : ℝ) (reg : ℕ → ℝ) :
    (fun n => operatorRadius K n α / reg n) =O[Filter.atTop]
      (fun n => 1 / (reg n * Real.sqrt (n : ℝ))) := by
  apply Asymptotics.IsBigO.of_bound ‖K * confidenceMultiplier α‖
  exact Filter.Eventually.of_forall fun n => by rw [operatorRadius_scaled, norm_mul]

/-- A finite set of positive values has a common positive lower bound.
The empty case is harmless and covers singleton families of regimes. -/
theorem finite_positive_margin {ι : Type*} (s : Finset ι) (D : ι → ℝ)
    (hD : ∀ i ∈ s, 0 < D i) : ∃ δ : ℝ, 0 < δ ∧ ∀ i ∈ s, δ ≤ D i := by
  classical
  induction s using Finset.induction_on with
  | empty => exact ⟨1, by norm_num, by simp⟩
  | @insert a s ha ih =>
    obtain ⟨δ, hδ, hbound⟩ := ih (fun i hi => hD i (Finset.mem_insert_of_mem hi))
    refine ⟨min (D a) δ, lt_min (hD a (Finset.mem_insert_self _ _)) hδ, ?_⟩
    intro i hi
    rcases Finset.mem_insert.mp hi with rfl | hi
    · exact min_le_left _ _
    · exact (min_le_right _ _).trans (hbound i hi)

namespace Statistical
variable {X E ι : Type*} [MeasurableSpace X] [Finite ι]
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

theorem populationScore_pos (P Q : Measure X)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {φ : X → E} {K reg : ℝ} (hφ : StronglyMeasurable φ)
    (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hreg : 0 < reg)
    (hne : Moments.mean Q φ ≠ Moments.mean P φ) :
    0 < Moments.populationScore P Q hφ hdiag reg hreg := by
  have hnonneg := Operators.quadraticScore_nonneg
    (populationOperator P hφ hdiag reg hreg) (population_positive P hφ hdiag hreg)
    (Moments.mean Q φ - Moments.mean P φ)
  have hzero := Operators.quadraticScore_eq_zero_iff
    (populationOperator P hφ hdiag reg hreg) hreg (population_coercive P hφ hdiag hreg)
    (Moments.mean Q φ - Moments.mean P φ)
  apply lt_of_le_of_ne hnonneg
  intro hz
  exact hne (sub_eq_zero.mp (hzero.mp hz.symm))

/-- Injective mean embeddings for a fixed finite bank give a positive common
score separation at fixed regularization. -/
theorem finite_family_separation (P : ι → Measure X)
    [∀ i, IsProbabilityMeasure (P i)] {φ : X → E} {K reg : ℝ}
    (hφ : StronglyMeasurable φ) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (hreg : 0 < reg)
    (hinjective : Function.Injective (fun i => Moments.mean (P i) φ)) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ i j, i ≠ j →
      δ ≤ Moments.populationScore (P i) (P j) hφ hdiag reg hreg := by
  classical
  let := Fintype.ofFinite ι
  let s : Finset (ι × ι) := Finset.univ.filter (fun p => p.1 ≠ p.2)
  obtain ⟨δ, hδ, hb⟩ := finite_positive_margin s
    (fun p => Moments.populationScore (P p.1) (P p.2) hφ hdiag reg hreg) (by
      intro p hp
      have hne : p.1 ≠ p.2 := (Finset.mem_filter.mp hp).2
      exact populationScore_pos _ _ hφ hdiag hreg
        (fun h => hne (hinjective h).symm))
  exact ⟨δ, hδ, fun i j hij => hb (i, j) (by simp [s, hij])⟩

end Statistical
end PearsonClassification
