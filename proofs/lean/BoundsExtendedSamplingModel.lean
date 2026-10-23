module

public import BoundsExtendedStatistical

/-! Identification of the recursive sampling law with the usual finite product law. -/

@[expose] public section
noncomputable section
namespace PearsonClassification.BoundedDifferences
open MeasureTheory ProbabilityTheory
variable {X : Type*} [MeasurableSpace X]

theorem measurable_observations (n : ℕ) : Measurable (observations (X := X) n) :=
  measurable_pi_iff.mpr (measurable_observation n)

theorem law_rectangle (n : ℕ) (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)] (s : Fin n → Set X)
    (hs : ∀ i, MeasurableSet (s i)) :
    law n μ (observations n ⁻¹' Set.pi Set.univ s) = ∏ i, μ i (s i) := by
  induction n with
  | zero => simp [law, Set.pi, observations]
  | succ n ih =>
    have heq : observations (n + 1) ⁻¹' Set.pi Set.univ s =
        (observations n ⁻¹' Set.pi Set.univ (fun i => s i.castSucc)) ×ˢ s (Fin.last n) := by
      ext p
      simp only [Set.mem_preimage, Set.mem_pi, Set.mem_univ, forall_const,
        Set.mem_prod, observations]
      constructor
      · intro h
        exact ⟨fun i => by simpa only [Fin.snoc_castSucc] using h i.castSucc,
          by simpa only [Fin.snoc_last] using h (Fin.last n)⟩
      · rintro ⟨h, hl⟩ i
        refine Fin.lastCases ?_ (fun j => ?_) i
        · simpa only [Fin.snoc_last] using hl
        · simpa only [Fin.snoc_castSucc] using h j
    rw [heq, law, Measure.prod_prod, ih _ _ (fun i => hs i.castSucc),
      Fin.prod_univ_castSucc]

theorem observations_map (n : ℕ) (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)] :
    (law n μ).map (observations n) = Measure.pi μ := by
  apply Eq.symm
  apply Measure.pi_eq
  intro s hs
  rw [Measure.map_apply (measurable_observations n)
    (MeasurableSet.pi Set.countable_univ (fun i _ => hs i))]
  exact law_rectangle n μ s hs

theorem observation_map (n : ℕ) (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)] (i : Fin n) :
    (law n μ).map (fun s => observations n s i) = μ i := by
  have h := (measurePreserving_eval μ i).map_eq
  rw [← observations_map n μ] at h
  rw [Measure.map_map (measurable_pi_apply i) (measurable_observations n)] at h
  exact h

theorem observations_independent (n : ℕ) (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)] :
    iIndepFun (fun i s => observations n s i) (law n μ) := by
  let : IsProbabilityMeasure (law n μ) := law_isProbability n μ (fun _ => inferInstance)
  apply (iIndepFun_iff_map_fun_eq_pi_map
    (fun i => (measurable_observation n i).aemeasurable)).2
  simp_rw [observation_map]
  exact observations_map n μ

/-- Encode the usual finite tuple as the recursive product sample. -/
def pack : (n : ℕ) → (Fin n → X) → Sample X n
  | 0, _ => PUnit.unit
  | n + 1, z => (pack n (fun i => z i.castSucc), z (Fin.last n))

omit [MeasurableSpace X] in
theorem observations_pack (n : ℕ) (z : Fin n → X) : observations n (pack n z) = z := by
  induction n with
  | zero => exact funext (fun i => Fin.elim0 i)
  | succ n ih =>
    funext i
    refine Fin.lastCases ?_ (fun j => ?_) i
    · simp only [pack, observations, Fin.snoc_last]
    · simp only [pack, observations, Fin.snoc_castSucc, ih]

omit [MeasurableSpace X] in
theorem pack_observations (n : ℕ) (s : Sample X n) : pack n (observations n s) = s := by
  induction n with
  | zero => exact Subsingleton.elim _ _
  | succ n ih =>
    simp only [pack, observations, Fin.snoc_castSucc, Fin.snoc_last, ih]

theorem measurable_pack (n : ℕ) : Measurable (pack (X := X) n) := by
  induction n with
  | zero => exact measurable_const
  | succ n ih =>
    exact (ih.comp (measurable_pi_iff.mpr fun i => measurable_pi_apply i.castSucc)).prodMk
      (measurable_pi_apply (Fin.last n))

theorem pack_map_pi (n : ℕ) (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)] :
    (Measure.pi μ).map (pack n) = law n μ := by
  have h := congrArg (Measure.map (pack n)) (observations_map n μ)
  rw [Measure.map_map (measurable_pack n) (measurable_observations n)] at h
  have hid : pack (X := X) n ∘ observations n = id := funext (pack_observations n)
  rw [hid, Measure.map_id] at h
  exact h.symm

/-- Any independent measurable observations with the stated marginals have
exactly the law used by the concentration and score theorems. -/
theorem independent_samples_law {Ω : Type*} [MeasurableSpace Ω]
    (ν : Measure Ω) [IsProbabilityMeasure ν] (n : ℕ)
    (μ : Fin n → Measure X) [∀ i, IsProbabilityMeasure (μ i)] (Z : Fin n → Ω → X)
    (hZ : ∀ i, Measurable (Z i)) (hind : iIndepFun Z ν)
    (hmarginal : ∀ i, ν.map (Z i) = μ i) :
    ν.map (fun ω => pack n (fun i => Z i ω)) = law n μ := by
  have h := hind.map_fun_eq_pi_map (fun i => (hZ i).aemeasurable)
  simp_rw [hmarginal] at h
  have ht : Measurable (fun ω i => Z i ω) := measurable_pi_iff.mpr hZ
  rw [← pack_map_pi n μ, ← h, Measure.map_map (measurable_pack n) ht]
  rfl

end PearsonClassification.BoundedDifferences
