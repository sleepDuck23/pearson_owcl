module

public import Mathlib

/-! Bounded-difference concentration, derived from Hoeffding's lemma. -/

@[expose] public section

noncomputable section

universe u

namespace PearsonClassification
namespace BoundedDifferences

open MeasureTheory ProbabilityTheory
open scoped NNReal

variable {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]

theorem integrable_of_bounded (μ : Measure X) [IsProbabilityMeasure μ]
    {f : X → ℝ} (hf : Measurable f) {M : ℝ} (hb : ∀ x, |f x| ≤ M) : Integrable f μ :=
  Integrable.of_bound hf.aestronglyMeasurable M (ae_of_all μ hb)

theorem subgaussian_of_diameter (μ : Measure X) [IsProbabilityMeasure μ]
    {f : X → ℝ} (hf : Measurable f) {M c : ℝ} (hb : ∀ x, |f x| ≤ M)
    (hc : 0 ≤ c) (hd : ∀ x y, |f x - f y| ≤ c) :
    HasSubgaussianMGF (fun x => f x - ∫ y, f y ∂μ) ((NNReal.mk c hc / 2 : ℝ≥0) ^ 2) μ := by
  let : Nonempty X := nonempty_of_isProbabilityMeasure μ
  let a := sInf (Set.range f)
  have hne : (Set.range f).Nonempty := Set.range_nonempty f
  have hbelow : BddBelow (Set.range f) := by
    refine ⟨-M, ?_⟩
    rintro _ ⟨x, rfl⟩
    exact (abs_le.1 (hb x)).1
  have hlo (x : X) : a ≤ f x := csInf_le hbelow (Set.mem_range_self x)
  have hhi (x : X) : f x ≤ a + c := by
    have h := le_csInf hne (show ∀ z ∈ Set.range f, f x - c ≤ z from by
      rintro _ ⟨y, rfl⟩
      have hxy := (abs_le.1 (hd x y)).2
      linarith)
    linarith
  have h := hasSubgaussianMGF_of_mem_Icc hf.aemeasurable
    (ae_of_all μ fun x => (show f x ∈ Set.Icc a (a + c) from ⟨hlo x, hhi x⟩))
  have hparam : ‖a + c - a‖₊ = (NNReal.mk c hc : ℝ≥0) := by
    apply NNReal.eq
    simp only [coe_nnnorm, Real.norm_eq_abs, add_sub_cancel_left, abs_of_nonneg hc,
      NNReal.coe_mk]
  rw [hparam] at h
  exact h

theorem integral_abs_le_bound (μ : Measure X) [IsProbabilityMeasure μ]
    {f : X → ℝ} (hf : Measurable f) {M : ℝ} (hb : ∀ x, |f x| ≤ M) :
    |∫ x, f x ∂μ| ≤ M := by
  calc
    |∫ x, f x ∂μ| ≤ ∫ x, |f x| ∂μ := abs_integral_le_integral_abs
    _ ≤ ∫ _ : X, M ∂μ :=
      integral_mono (integrable_of_bounded μ hf hb).abs (integrable_const _) hb
    _ = M := by simp

/-- The induction step: integrate one independent coordinate and add its
Hoeffding parameter. The coordinate may depend on the remaining coordinates
through `f`; no independence of the resulting summands is assumed. -/
theorem product_step (μ : Measure X) (ν : Measure Y)
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    {f : X × Y → ℝ} (hf : Measurable f) {M c : ℝ}
    (hb : ∀ p, |f p| ≤ M) (hc : 0 ≤ c)
    (hd : ∀ x y z, |f (x, y) - f (x, z)| ≤ c)
    {C : ℝ≥0}
    (hg : HasSubgaussianMGF
      (fun x => (∫ y, f (x, y) ∂ν) - ∫ z, (∫ y, f (z, y) ∂ν) ∂μ) C μ) :
    HasSubgaussianMGF (fun p => f p - ∫ q, f q ∂μ.prod ν)
      (C + (NNReal.mk c hc / 2 : ℝ≥0) ^ 2) (μ.prod ν) := by
  let g : X → ℝ := fun x => ∫ y, f (x, y) ∂ν
  have hfm (x : X) : Measurable (fun y => f (x, y)) :=
    hf.comp (measurable_const.prodMk measurable_id)
  have hgm : Measurable g := hf.stronglyMeasurable.integral_prod_right'.measurable
  have hfi := integrable_of_bounded (μ.prod ν) hf hb
  have heq : (∫ p, f p ∂μ.prod ν) = ∫ x, g x ∂μ := integral_prod f hfi
  have hcenter (p : X × Y) : |f p - ∫ q, f q ∂μ.prod ν| ≤
      M + |∫ q, f q ∂μ.prod ν| :=
    (abs_sub _ _).trans (add_le_add (hb p) le_rfl)
  have hexp (t : ℝ) : Integrable (fun p => Real.exp (t *
      (f p - ∫ q, f q ∂μ.prod ν))) (μ.prod ν) := by
    apply integrable_exp_mul_of_mem_Icc (hf.sub_const _).aemeasurable
    exact ae_of_all _ fun p => abs_le.1 (hcenter p)
  refine ⟨hexp, ?_⟩
  intro t
  have hrepr : mgf (fun p => f p - ∫ q, f q ∂μ.prod ν) (μ.prod ν) t =
      ∫ x, Real.exp (t * (g x - ∫ z, g z ∂μ)) *
        mgf (fun y => f (x, y) - g x) ν t ∂μ := by
    rw [mgf, integral_prod _ (hexp t), heq]
    apply integral_congr_ae
    apply ae_of_all μ
    intro x
    have hid (y : Y) : t * (f (x, y) - ∫ z, g z ∂μ) =
        t * (g x - ∫ z, g z ∂μ) + t * (f (x, y) - g x) := by ring
    simp_rw [hid, Real.exp_add]
    exact integral_const_mul _ _
  rw [hrepr]
  calc
    (∫ x, Real.exp (t * (g x - ∫ z, g z ∂μ)) *
        mgf (fun y => f (x, y) - g x) ν t ∂μ) ≤
        ∫ x, Real.exp (t * (g x - ∫ z, g z ∂μ)) *
          Real.exp (((NNReal.mk c hc / 2 : ℝ≥0) ^ 2 : ℝ) * t ^ 2 / 2) ∂μ := by
      apply integral_mono_of_nonneg
      · exact ae_of_all μ fun x => mul_nonneg (Real.exp_pos _).le
          (integral_nonneg fun _ => (Real.exp_pos _).le)
      · exact (hg.integrable_exp_mul t).mul_const _
      · apply ae_of_all μ
        intro x
        exact mul_le_mul_of_nonneg_left
          ((subgaussian_of_diameter ν (hfm x) (fun y => hb (x, y)) hc (hd x)).mgf_le t)
          (Real.exp_pos _).le
    _ = mgf (fun x => g x - ∫ z, g z ∂μ) μ t *
        Real.exp (((NNReal.mk c hc / 2 : ℝ≥0) ^ 2 : ℝ) * t ^ 2 / 2) := integral_mul_const _ _
    _ ≤ Real.exp (C * t ^ 2 / 2) *
        Real.exp (((NNReal.mk c hc / 2 : ℝ≥0) ^ 2 : ℝ) * t ^ 2 / 2) :=
      mul_le_mul_of_nonneg_right (hg.mgf_le t) (Real.exp_pos _).le
    _ = Real.exp ((↑(C + (NNReal.mk c hc / 2 : ℝ≥0) ^ 2) : ℝ) * t ^ 2 / 2) := by
      rw [← Real.exp_add, NNReal.coe_add, NNReal.coe_pow]
      congr 1
      ring

/-! ## Canonical finite independent sampling space -/

@[reducible] def Sample (X : Type u) : ℕ → Type u
  | 0 => PUnit
  | n + 1 => Sample X n × X

@[instance_reducible] def sampleMeasurable : (n : ℕ) → MeasurableSpace (Sample X n)
  | 0 => inferInstanceAs (MeasurableSpace PUnit)
  | n + 1 => by
    letI := sampleMeasurable n
    exact inferInstanceAs (MeasurableSpace (Sample X n × X))

instance (n : ℕ) : MeasurableSpace (Sample X n) := sampleMeasurable n

def law : (n : ℕ) → (Fin n → Measure X) → Measure (Sample X n)
  | 0, _ => Measure.dirac PUnit.unit
  | n + 1, μ => (law n (fun i => μ i.castSucc)).prod (μ (Fin.last n))

theorem law_isProbability (n : ℕ) (μ : Fin n → Measure X)
    (hμ : ∀ i, IsProbabilityMeasure (μ i)) : IsProbabilityMeasure (law n μ) := by
  induction n with
  | zero => exact inferInstanceAs (IsProbabilityMeasure (Measure.dirac PUnit.unit))
  | succ n ih =>
    let := ih (fun i => μ i.castSucc) (fun i => hμ i.castSucc)
    let := hμ (Fin.last n)
    exact inferInstanceAs (IsProbabilityMeasure
      ((law n (fun i => μ i.castSucc)).prod (μ (Fin.last n))))

instance (n : ℕ) (μ : Fin n → Measure X) [∀ i, IsProbabilityMeasure (μ i)] :
    IsProbabilityMeasure (law n μ) := law_isProbability n μ (fun _ => inferInstance)

def observations : (n : ℕ) → Sample X n → Fin n → X
  | 0, _ => Fin.elim0
  | n + 1, p => Fin.snoc (observations n p.1) p.2

theorem measurable_observation (n : ℕ) (i : Fin n) :
    Measurable (fun s : Sample X n => observations n s i) := by
  induction n with
  | zero => exact Fin.elim0 i
  | succ n ih =>
    refine Fin.lastCases ?_ (fun j => ?_) i
    · simpa only [observations, Fin.snoc_last, instMeasurableSpaceSample, sampleMeasurable]
        using (measurable_snd : Measurable (fun s : Sample X n × X => s.2))
    · simpa only [observations, Fin.snoc_castSucc, Function.comp_def,
        instMeasurableSpaceSample, sampleMeasurable] using (ih j).comp measurable_fst

/-- Coordinate sensitivity, expressed on the full independent sample. -/
def HasDifferences {n : ℕ} (f : Sample X n → ℝ) (c : Fin n → ℝ≥0) : Prop :=
  ∀ i s t, (∀ j, j ≠ i → observations n s j = observations n t j) →
    |f s - f t| ≤ c i

omit [MeasurableSpace X] in
theorem hasDifferences_last {n : ℕ} {f : Sample X (n + 1) → ℝ}
    {c : Fin (n + 1) → ℝ≥0} (hd : HasDifferences f c) (s : Sample X n) (x y : X) :
    |f (s, x) - f (s, y)| ≤ c (Fin.last n) := by
  apply hd (Fin.last n) (s, x) (s, y)
  intro j
  refine Fin.lastCases ?_ (fun k => ?_) j
  · intro hj
    exact False.elim (hj rfl)
  · intro _
    simp only [observations, Fin.snoc_castSucc]

theorem hasDifferences_integral {n : ℕ} (ν : Measure X) [IsProbabilityMeasure ν]
    {f : Sample X (n + 1) → ℝ} {c : Fin (n + 1) → ℝ≥0}
    (hf : Measurable f) {M : ℝ} (hb : ∀ s, |f s| ≤ M) (hd : HasDifferences f c) :
    HasDifferences (fun s => ∫ x, f (s, x) ∂ν) (fun i => c i.castSucc) := by
  intro i s t hst
  have hfm (s : Sample X n) : Measurable (fun x => f (s, x)) :=
    hf.comp (measurable_const.prodMk measurable_id)
  have hchange (x : X) : |f (s, x) - f (t, x)| ≤ c i.castSucc := by
    apply hd i.castSucc (s, x) (t, x)
    intro j
    refine Fin.lastCases ?_ (fun k => ?_) j
    · intro _
      simp only [observations, Fin.snoc_last]
    · intro hj
      simp only [observations, Fin.snoc_castSucc]
      apply hst k
      intro hki
      exact hj (congrArg Fin.castSucc hki)
  have hdm : Measurable (fun x => f (s, x) - f (t, x)) :=
    (hfm s).sub (hfm t)
  have h := integral_abs_le_bound (f := fun x => f (s, x) - f (t, x)) ν hdm hchange
  rwa [integral_sub (integrable_of_bounded ν (hfm s) (fun x => hb (s, x)))
    (integrable_of_bounded ν (hfm t) (fun x => hb (t, x)))] at h

/-- McDiarmid's exponential-moment estimate for a finite independent sample.
Boundedness is sufficient for the applications here and avoids integrability
assumptions about an arbitrary measurable statistic. -/
theorem subgaussian_of_hasDifferences (n : ℕ) (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)]
    {f : Sample X n → ℝ} (hf : Measurable f) {M : ℝ}
    (hb : ∀ s, |f s| ≤ M) (c : Fin n → ℝ≥0) (hd : HasDifferences f c) :
    HasSubgaussianMGF (fun s => f s - ∫ t, f t ∂law n μ)
      (∑ i, (c i / 2) ^ 2) (law n μ) := by
  induction n with
  | zero =>
    let : IsProbabilityMeasure (law 0 μ) := law_isProbability 0 μ (fun i => Fin.elim0 i)
    have hid : (fun s => f s - ∫ t, f t ∂law 0 μ) = (fun _ => 0) := by
      funext s
      have hs : s = PUnit.unit := Subsingleton.elim _ _
      simp only [law, integral_dirac, hs, sub_self]
    rw [hid]
    simpa only [Fin.sum_univ_zero] using HasSubgaussianMGF.fun_zero (μ := law 0 μ)
  | succ n ih =>
    let g : Sample X n → ℝ := fun s => ∫ x, f (s, x) ∂μ (Fin.last n)
    have hfm (s : Sample X n) : Measurable (fun x => f (s, x)) :=
      hf.comp (measurable_const.prodMk measurable_id)
    have hgm : Measurable g := hf.stronglyMeasurable.integral_prod_right'.measurable
    have hgb (s : Sample X n) : |g s| ≤ M :=
      integral_abs_le_bound (μ (Fin.last n)) (hfm s) (fun x => hb (s, x))
    have hgd := hasDifferences_integral (μ (Fin.last n)) hf hb hd
    have hmg := ih (fun i => μ i.castSucc) hgm hgb (fun i => c i.castSucc) hgd
    have hstep := product_step (law n (fun i => μ i.castSucc)) (μ (Fin.last n))
      hf hb (c (Fin.last n)).coe_nonneg (hasDifferences_last hd) hmg
    simpa only [law, Fin.sum_univ_castSucc, NNReal.mk_coe] using hstep

theorem mcdiarmid_tail (n : ℕ) (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)]
    {f : Sample X n → ℝ} (hf : Measurable f) {M : ℝ}
    (hb : ∀ s, |f s| ≤ M) (c : Fin n → ℝ≥0) (hd : HasDifferences f c)
    {t : ℝ} (ht : 0 ≤ t) :
    (law n μ).real {s | t ≤ f s - ∫ q, f q ∂law n μ} ≤
      Real.exp (-2 * t ^ 2 / ∑ i, (c i : ℝ) ^ 2) := by
  have h := (subgaussian_of_hasDifferences n μ hf hb c hd).measure_ge_le ht
  have hcoef : (2 : ℝ) * (↑(∑ i, (c i / 2) ^ 2 : ℝ≥0) : ℝ) =
      (∑ i, (c i : ℝ) ^ 2) / 2 := by
    simp only [NNReal.coe_sum, NNReal.coe_pow, NNReal.coe_div, NNReal.coe_ofNat]
    rw [Finset.sum_div]
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hcoef] at h
  convert h using 1
  congr 1
  ring

omit [MeasurableSpace X] in
/-- A finite coordinate-sensitivity bound controls the global diameter. -/
theorem global_diameter (n : ℕ) {f : Sample X n → ℝ} {c : Fin n → ℝ≥0}
    (hd : HasDifferences f c) (s t : Sample X n) :
    |f s - f t| ≤ ∑ i, (c i : ℝ) := by
  induction n with
  | zero =>
    have heq : s = t := Subsingleton.elim _ _
    simp [heq]
  | succ n ih =>
    have hslice : HasDifferences (fun r => f (r, s.2)) (fun i => c i.castSucc) := by
      intro i u v huv
      apply hd i.castSucc (u, s.2) (v, s.2)
      intro j
      refine Fin.lastCases ?_ (fun k => ?_) j
      · intro _
        simp only [observations, Fin.snoc_last]
      · intro hj
        simp only [observations, Fin.snoc_castSucc]
        exact huv k (fun hki => hj (congrArg Fin.castSucc hki))
    calc
      |f s - f t| ≤ |f s - f (t.1, s.2)| + |f (t.1, s.2) - f t| :=
        abs_sub_le _ _ _
      _ ≤ (∑ i : Fin n, (c i.castSucc : ℝ)) + c (Fin.last n) :=
        add_le_add (ih hslice s.1 t.1) (hasDifferences_last hd t.1 s.2 t.2)
      _ = ∑ i, (c i : ℝ) := (Fin.sum_univ_castSucc (fun i => (c i : ℝ))).symm

/-- The manuscript's McDiarmid theorem; boundedness follows from the finite
coordinate bounds and is not an additional analytic premise. -/
theorem mcdiarmid (n : ℕ) (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)] {f : Sample X n → ℝ}
    (hf : Measurable f) (c : Fin n → ℝ≥0) (hd : HasDifferences f c)
    {t : ℝ} (ht : 0 ≤ t) :
    (law n μ).real {s | t ≤ f s - ∫ q, f q ∂law n μ} ≤
      Real.exp (-2 * t ^ 2 / ∑ i, (c i : ℝ) ^ 2) := by
  let : IsProbabilityMeasure (law n μ) := law_isProbability n μ (fun _ => inferInstance)
  have : Nonempty (Sample X n) := nonempty_of_isProbabilityMeasure (law n μ)
  let a : Sample X n := Classical.choice inferInstance
  apply mcdiarmid_tail n μ hf (M := |f a| + ∑ i, (c i : ℝ)) _ c hd ht
  intro s
  calc
    |f s| ≤ |f s - f a| + |f a| := by simpa using abs_add_le (f s - f a) (f a)
    _ ≤ (∑ i, (c i : ℝ)) + |f a| := add_le_add (global_diameter n hd s a) le_rfl
    _ = |f a| + ∑ i, (c i : ℝ) := add_comm _ _

theorem confidence_of_subgaussian (μ : Measure X) [IsProbabilityMeasure μ]
    {f : X → ℝ} {C : ℝ≥0}
    (hmg : HasSubgaussianMGF (fun x => f x - ∫ y, f y ∂μ) C μ)
    (hC : 0 < C) {γ : ℝ} (hγ : 0 < γ) (hγ₁ : γ < 1) :
    μ {x | (∫ y, f y ∂μ) + Real.sqrt (2 * C * Real.log (1 / γ)) < f x} ≤
      ENNReal.ofReal γ := by
  have hCR : 0 < (C : ℝ) := hC
  have hlog : 0 ≤ Real.log (1 / γ) :=
    Real.log_nonneg ((le_div_iff₀ hγ).2 (by linarith))
  let t := Real.sqrt (2 * C * Real.log (1 / γ))
  have htail := hmg.measure_ge_le (Real.sqrt_nonneg (2 * C * Real.log (1 / γ)))
  have hexp : Real.exp (-t ^ 2 / (2 * C)) = γ := by
    dsimp only [t]
    rw [Real.sq_sqrt (by positivity)]
    have hid : -(2 * (C : ℝ) * Real.log (1 / γ)) / (2 * C) =
        -Real.log (1 / γ) := by field_simp
    rw [hid, one_div, Real.log_inv, neg_neg, Real.exp_log hγ]
  have hsubset : {x | (∫ y, f y ∂μ) + t < f x} ⊆
      {x | t ≤ f x - ∫ y, f y ∂μ} := by
    intro x hx
    change (∫ y, f y ∂μ) + t < f x at hx
    change t ≤ f x - ∫ y, f y ∂μ
    linarith
  have hreal : μ.real {x | (∫ y, f y ∂μ) + t < f x} ≤ γ :=
    (measureReal_mono hsubset).trans (htail.trans_eq hexp)
  calc
    μ {x | (∫ y, f y ∂μ) + t < f x} =
        ENNReal.ofReal (μ.real {x | (∫ y, f y ∂μ) + t < f x}) :=
      (ofReal_measureReal).symm
    _ ≤ ENNReal.ofReal γ := ENNReal.ofReal_le_ofReal hreal

end BoundedDifferences
end PearsonClassification
