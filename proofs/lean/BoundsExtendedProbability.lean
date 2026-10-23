module

public import BoundsExtendedCore

@[expose] public section

noncomputable section

namespace PearsonClassification

/-!
## Statistical conclusions conditional on concentration

The hypotheses here are inequalities for the actual mean and operator error
events, not arbitrary unnamed good events. `BoundsExtendedStatistical`
discharges these hypotheses from independent sampling using McDiarmid.
-/

section CalibratedProbability

open MeasureTheory Operators
open scoped ENNReal RealInnerProductSpace

variable {Ω E : Type*} [MeasurableSpace Ω]
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E]

lemma probability_lower_of_complement (μ : Measure Ω) [IsProbabilityMeasure μ]
    (event : Set Ω) {α : ℝ≥0∞} (hbad : μ eventᶜ ≤ α) : 1 - α ≤ μ event := by
  apply tsub_le_iff_right.2
  have hcover : (1 : ℝ≥0∞) ≤ μ event + μ eventᶜ := by
    simpa only [Set.union_compl_self, measure_univ] using measure_union_le (μ := μ) event eventᶜ
  exact hcover.trans (add_le_add le_rfl hbad)

theorem null_error_of_concentration
    (μ : Measure Ω) (B : Ω → E ≃L[ℝ] E) (Δ : E) (Δ_hat : Ω → E)
    {K reg α η : ℝ} {n : ℕ}
    (hreg : 0 < reg) (hnull : Δ = 0)
    (hcoercive : ∀ ω f, reg * ‖f‖ ^ 2 ≤ ⟪f, B ω f⟫)
    (hmean : μ {ω | meanRadius K n α < ‖Δ_hat ω - Δ‖} ≤ ENNReal.ofReal α)
    (hthreshold : nullBound K n reg α ≤ η) :
    μ {ω | η < quadraticScore (B ω) (Δ_hat ω)} ≤ ENNReal.ofReal α := by
  apply le_trans (false_separation_measure_bound μ
    (fun ω => quadraticScore (B ω) (Δ_hat ω))
    {ω | ‖Δ_hat ω - Δ‖ ≤ meanRadius K n α} hthreshold ?_) ?_
  · intro ω hgood
    exact null_quadratic_upper_bound (B ω) Δ (Δ_hat ω) hreg hnull hgood (hcoercive ω)
  · simpa only [Set.compl_ofPred, not_le] using hmean

theorem alternative_error_of_concentration
    (μ : Measure Ω) (A : E ≃L[ℝ] E) (B : Ω → E ≃L[ℝ] E)
    (Δ : E) (Δ_hat : Ω → E) {δ K reg α η : ℝ} {n : ℕ}
    (hA : A.toContinuousLinearMap.IsPositive)
    (hB : ∀ ω, (B ω).toContinuousLinearMap.IsPositive)
    (hδ : 0 ≤ δ) (hK : 0 ≤ K) (hreg : 0 < reg) (hα : 0 ≤ α)
    (hseparation : δ ≤ quadraticScore A Δ)
    (hcoercive : ∀ f, reg * ‖f‖ ^ 2 ≤ ⟪f, A f⟫)
    (hmean : μ {ω | meanRadius K n (α / 2) < ‖Δ_hat ω - Δ‖} ≤ ENNReal.ofReal (α / 2))
    (hop : μ {ω | operatorRadius K n (α / 2) <
      ‖(B ω).toContinuousLinearMap - A.toContinuousLinearMap‖} ≤ ENNReal.ofReal (α / 2))
    (hthreshold : η < alternativeBound δ K n reg α) :
    μ {ω | quadraticScore (B ω) (Δ_hat ω) ≤ η} ≤ ENNReal.ofReal α := by
  have hmean' : μ {ω | ‖Δ_hat ω - Δ‖ ≤ meanRadius K n (α / 2)}ᶜ ≤
      ENNReal.ofReal (α / 2) := by simpa only [Set.compl_ofPred, not_le] using hmean
  have hop' : μ {ω | ‖(B ω).toContinuousLinearMap - A.toContinuousLinearMap‖ ≤
      operatorRadius K n (α / 2)}ᶜ ≤ ENNReal.ofReal (α / 2) := by
    simpa only [Set.compl_ofPred, not_le] using hop
  have h := false_merging_confidence_bound μ (fun ω => quadraticScore (B ω) (Δ_hat ω))
    {ω | ‖Δ_hat ω - Δ‖ ≤ meanRadius K n (α / 2)}
    {ω | ‖(B ω).toContinuousLinearMap - A.toContinuousLinearMap‖ ≤ operatorRadius K n (α / 2)}
    hthreshold (fun ω hm ho => nonmerging_quadratic_lower_bound A (B ω) Δ (Δ_hat ω)
      hA (hB ω) hδ hreg (operatorRadius_nonneg hK) hseparation hm hcoercive ho)
    hmean' hop'
  have hhalf : ENNReal.ofReal (α / 2) + ENNReal.ofReal (α / 2) = ENNReal.ofReal α := by
    rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
    congr 1
    ring
  simpa only [hhalf] using h

theorem null_high_probability_of_concentration
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (B : Ω → E ≃L[ℝ] E) (Δ : E) (Δ_hat : Ω → E) {K reg α : ℝ} {n : ℕ}
    (hB : ∀ ω, (B ω).toContinuousLinearMap.IsPositive)
    (hreg : 0 < reg) (hnull : Δ = 0)
    (hcoercive : ∀ ω f, reg * ‖f‖ ^ 2 ≤ ⟪f, B ω f⟫)
    (hmean : μ {ω | meanRadius K n α < ‖Δ_hat ω - Δ‖} ≤ ENNReal.ofReal α) :
    1 - ENNReal.ofReal α ≤ μ {ω | 0 ≤ quadraticScore (B ω) (Δ_hat ω) ∧
      quadraticScore (B ω) (Δ_hat ω) ≤ nullBound K n reg α} := by
  apply probability_lower_of_complement
  have hbad := null_error_of_concentration μ B Δ Δ_hat hreg hnull hcoercive hmean le_rfl
  apply le_trans (measure_mono ?_) hbad
  intro ω hfailure
  have hn := quadraticScore_nonneg (B ω) (hB ω) (Δ_hat ω)
  exact lt_of_not_ge (fun hu => hfailure ⟨hn, hu⟩)

theorem alternative_high_probability_of_concentration
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (A : E ≃L[ℝ] E) (B : Ω → E ≃L[ℝ] E) (Δ : E) (Δ_hat : Ω → E)
    {δ K reg α : ℝ} {n : ℕ}
    (hA : A.toContinuousLinearMap.IsPositive)
    (hB : ∀ ω, (B ω).toContinuousLinearMap.IsPositive)
    (hδ : 0 ≤ δ) (hK : 0 ≤ K) (hreg : 0 < reg) (hα : 0 ≤ α)
    (hseparation : δ ≤ quadraticScore A Δ)
    (hcoercive : ∀ f, reg * ‖f‖ ^ 2 ≤ ⟪f, A f⟫)
    (hmean : μ {ω | meanRadius K n (α / 2) < ‖Δ_hat ω - Δ‖} ≤ ENNReal.ofReal (α / 2))
    (hop : μ {ω | operatorRadius K n (α / 2) <
      ‖(B ω).toContinuousLinearMap - A.toContinuousLinearMap‖} ≤ ENNReal.ofReal (α / 2)) :
    1 - ENNReal.ofReal α ≤ μ {ω | alternativeBound δ K n reg α ≤
      quadraticScore (B ω) (Δ_hat ω)} := by
  apply probability_lower_of_complement
  calc
    μ {ω | alternativeBound δ K n reg α ≤ quadraticScore (B ω) (Δ_hat ω)}ᶜ ≤
        μ {ω | meanRadius K n (α / 2) < ‖Δ_hat ω - Δ‖} +
          μ {ω | operatorRadius K n (α / 2) <
            ‖(B ω).toContinuousLinearMap - A.toContinuousLinearMap‖} := by
      apply le_trans (measure_mono ?_) (measure_union_le _ _)
      intro ω hbad
      by_cases hm : ‖Δ_hat ω - Δ‖ ≤ meanRadius K n (α / 2)
      · right
        by_contra ho
        have ho' : ‖(B ω).toContinuousLinearMap - A.toContinuousLinearMap‖ ≤
            operatorRadius K n (α / 2) := le_of_not_gt ho
        exact hbad (nonmerging_quadratic_lower_bound A (B ω) Δ (Δ_hat ω)
          hA (hB ω) hδ hreg (operatorRadius_nonneg hK) hseparation hm hcoercive ho')
      · exact Or.inl (lt_of_not_ge hm)
    _ ≤ ENNReal.ofReal (α / 2) + ENNReal.ofReal (α / 2) := add_le_add hmean hop
    _ = ENNReal.ofReal α := by
      rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
      congr 1
      ring

end CalibratedProbability

/-! ## A fixed finite memory bank, with no independence between scores -/

section MemoryBank

open MeasureTheory
open scoped ENNReal

variable {Ω ι : Type*} [MeasurableSpace Ω] [Fintype ι]

def acceptedReferences (S : ι → Ω → ℝ) (η : ℝ) (ω : Ω) : Set ι :=
  {i | S i ω ≤ η}

theorem bank_unknown_failure_bound (μ : Measure Ω) (S : ι → Ω → ℝ)
    (η : ℝ) (ε : ℝ≥0∞)
    (herror : ∀ i, μ {ω | S i ω ≤ η} ≤ ε) :
    μ {ω | acceptedReferences S η ω ≠ ∅} ≤ Fintype.card ι * ε := by
  calc
    μ {ω | acceptedReferences S η ω ≠ ∅} ≤ μ (⋃ i, {ω | S i ω ≤ η}) := by
      apply measure_mono
      intro ω hbad
      obtain ⟨i, hi⟩ := Set.nonempty_iff_ne_empty.2 hbad
      exact Set.mem_iUnion.2 ⟨i, hi⟩
    _ ≤ ∑ i, μ {ω | S i ω ≤ η} := measure_iUnion_fintype_le μ _
    _ ≤ ∑ _ : ι, ε := Finset.sum_le_sum fun i _ => herror i
    _ = Fintype.card ι * ε := by simp

theorem bank_known_failure_bound (μ : Measure Ω) (S : ι → Ω → ℝ)
    (η : ℝ) (j : ι) (ε : ℝ≥0∞)
    (hnull : μ {ω | η < S j ω} ≤ ε)
    (halternative : ∀ i, i ≠ j → μ {ω | S i ω ≤ η} ≤ ε) :
    μ {ω | acceptedReferences S η ω ≠ {j}} ≤ Fintype.card ι * ε := by
  classical
  let failure : ι → Set Ω := fun i => if i = j then {ω | η < S i ω} else {ω | S i ω ≤ η}
  have hfailure : ∀ i, μ (failure i) ≤ ε := by
    intro i
    by_cases hij : i = j
    · simpa [failure, hij] using hnull
    · simpa [failure, hij] using halternative i hij
  calc
    μ {ω | acceptedReferences S η ω ≠ {j}} ≤ μ (⋃ i, failure i) := by
      apply measure_mono
      intro ω hbad
      by_contra hnone
      have hall : ∀ i, ω ∉ failure i := by simpa only [Set.mem_iUnion, not_exists] using hnone
      apply hbad
      ext i
      by_cases hij : i = j
      · subst i
        have hj := hall j
        simpa [acceptedReferences, failure] using hj
      · have hi := hall i
        simp only [failure, ite_eq_right hij, Set.mem_ofPred_eq] at hi
        simp [acceptedReferences, hij, hi]
    _ ≤ ∑ i, μ (failure i) := measure_iUnion_fintype_le μ _
    _ ≤ ∑ _ : ι, ε := Finset.sum_le_sum fun i _ => hfailure i
    _ = Fintype.card ι * ε := by simp

theorem bank_unknown_high_probability (μ : Measure Ω) [IsProbabilityMeasure μ]
    (S : ι → Ω → ℝ) (η : ℝ) (ε : ℝ≥0∞)
    (herror : ∀ i, μ {ω | S i ω ≤ η} ≤ ε) :
    1 - Fintype.card ι * ε ≤ μ {ω | acceptedReferences S η ω = ∅} := by
  apply probability_lower_of_complement
  simpa only [Set.compl_ofPred] using bank_unknown_failure_bound μ S η ε herror

theorem bank_known_high_probability (μ : Measure Ω) [IsProbabilityMeasure μ]
    (S : ι → Ω → ℝ) (η : ℝ) (j : ι) (ε : ℝ≥0∞)
    (hnull : μ {ω | η < S j ω} ≤ ε)
    (halternative : ∀ i, i ≠ j → μ {ω | S i ω ≤ η} ≤ ε) :
    1 - Fintype.card ι * ε ≤ μ {ω | acceptedReferences S η ω = {j}} := by
  apply probability_lower_of_complement
  simpa only [Set.compl_ofPred] using bank_known_failure_bound μ S η j ε hnull halternative

lemma bank_budget_split [Nonempty ι] {α : ℝ} (_hα : 0 ≤ α) :
    (Fintype.card ι : ℝ≥0∞) * ENNReal.ofReal (α / (Fintype.card ι : ℝ)) =
      ENNReal.ofReal α := by
  have hcard : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos
  rw [← ENNReal.ofReal_natCast (Fintype.card ι), ← ENNReal.ofReal_mul (by positivity)]
  congr 1
  field_simp

theorem bank_unknown_calibrated [Nonempty ι] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (S : ι → Ω → ℝ) (η : ℝ) {α : ℝ} (hα : 0 ≤ α)
    (herror : ∀ i, μ {ω | S i ω ≤ η} ≤ ENNReal.ofReal (α / (Fintype.card ι : ℝ))) :
    1 - ENNReal.ofReal α ≤ μ {ω | acceptedReferences S η ω = ∅} := by
  simpa only [bank_budget_split hα] using bank_unknown_high_probability μ S η _ herror

theorem bank_known_calibrated [Nonempty ι] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (S : ι → Ω → ℝ) (η : ℝ) (j : ι) {α : ℝ} (hα : 0 ≤ α)
    (hnull : μ {ω | η < S j ω} ≤ ENNReal.ofReal (α / (Fintype.card ι : ℝ)))
    (halternative : ∀ i, i ≠ j → μ {ω | S i ω ≤ η} ≤
      ENNReal.ofReal (α / (Fintype.card ι : ℝ))) :
    1 - ENNReal.ofReal α ≤ μ {ω | acceptedReferences S η ω = {j}} := by
  simpa only [bank_budget_split hα] using bank_known_high_probability μ S η j _ hnull halternative

end MemoryBank

end PearsonClassification
