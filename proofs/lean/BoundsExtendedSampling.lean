module

public import BoundsExtendedMcDiarmid
public import BoundsExtendedTensor

/-! Hilbert-space statistics of the actual independent observations. -/

@[expose] public section

noncomputable section

namespace PearsonClassification
namespace Sampling

open MeasureTheory InnerProductSpace BoundedDifferences
open scoped RealInnerProductSpace NNReal

variable {X E : Type*} [MeasurableSpace X]
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

def sumStat {n : ℕ} (v : Fin n → X → E) (s : Sample X n) : E :=
  ∑ i, v i (observations n s i)

omit [InnerProductSpace ℝ E] [CompleteSpace E] in
theorem sumStat_measurable {n : ℕ} {v : Fin n → X → E}
    (hv : ∀ i, StronglyMeasurable (v i)) : StronglyMeasurable (sumStat v) := by
  exact Finset.stronglyMeasurable_fun_sum _ fun i _ =>
    (hv i).comp_measurable (measurable_observation n i)

omit [MeasurableSpace X] [InnerProductSpace ℝ E] [CompleteSpace E] in
theorem sumStat_bound {n : ℕ} {v : Fin n → X → E} {B : Fin n → ℝ≥0}
    (hb : ∀ i x, ‖v i x‖ ≤ B i) (s : Sample X n) :
    ‖sumStat v s‖ ≤ ∑ i, (B i : ℝ) :=
  (norm_sum_le _ _).trans (Finset.sum_le_sum fun i _ => hb i _)

omit [InnerProductSpace ℝ E] [CompleteSpace E] in
theorem bounded_integrable (μ : Measure X) [IsProbabilityMeasure μ]
    {v : X → E} (hv : StronglyMeasurable v) {B : ℝ≥0} (hb : ∀ x, ‖v x‖ ≤ B) :
    Integrable v μ := Integrable.of_bound hv.aestronglyMeasurable B (ae_of_all μ hb)

omit [InnerProductSpace ℝ E] [CompleteSpace E] in
theorem bounded_square_integrable (μ : Measure X) [IsProbabilityMeasure μ]
    {v : X → E} (hv : StronglyMeasurable v) {B : ℝ≥0} (hb : ∀ x, ‖v x‖ ≤ B) :
    Integrable (fun x => ‖v x‖ ^ 2) μ := by
  exact Moments.normSquare_integrable μ hv (K := (B : ℝ) ^ 2)
    (fun x => (sq_le_sq₀ (norm_nonneg _) B.coe_nonneg).2 (hb x))

theorem integral_norm_add_sq (μ : Measure X) [IsProbabilityMeasure μ]
    {v : X → E} (hv : StronglyMeasurable v) {B : ℝ≥0} (hb : ∀ x, ‖v x‖ ≤ B)
    (hcenter : ∫ x, v x ∂μ = 0) (u : E) :
    (∫ x, ‖u + v x‖ ^ 2 ∂μ) = ‖u‖ ^ 2 + ∫ x, ‖v x‖ ^ 2 ∂μ := by
  have hi := bounded_integrable μ hv hb
  have hsq := bounded_square_integrable μ hv hb
  have hinner := (hi.const_inner u).const_mul (2 : ℝ)
  have hcross : Integrable (fun x => ‖u‖ ^ 2 + 2 * ⟪u, v x⟫) μ :=
    Integrable.add (integrable_const _) hinner
  simp_rw [norm_add_sq_real]
  rw [integral_add hcross hsq, integral_add (integrable_const _) hinner,
    integral_const_mul, integral_inner hi, hcenter]
  simp

theorem sumStat_variance (n : ℕ) (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)]
    (v : Fin n → X → E) (hv : ∀ i, StronglyMeasurable (v i))
    (B : Fin n → ℝ≥0) (hb : ∀ i x, ‖v i x‖ ≤ B i)
    (hcenter : ∀ i, ∫ x, v i x ∂μ i = 0) :
    (∫ s, ‖sumStat v s‖ ^ 2 ∂law n μ) = ∑ i, ∫ x, ‖v i x‖ ^ 2 ∂μ i := by
  induction n with
  | zero => simp [sumStat, law]
  | succ n ih =>
    let v' : Fin n → X → E := fun i => v i.castSucc
    let μ' : Fin n → Measure X := fun i => μ i.castSucc
    let B' : Fin n → ℝ≥0 := fun i => B i.castSucc
    let : IsProbabilityMeasure (law (n + 1) μ) := law_isProbability _ μ (fun _ => inferInstance)
    let : IsProbabilityMeasure (law n μ') := law_isProbability _ μ' (fun _ => inferInstance)
    have hsm := sumStat_measurable hv
    have hbound := sumStat_bound hb
    have hsqi : Integrable (fun s => ‖sumStat v s‖ ^ 2) (law (n + 1) μ) :=
      bounded_square_integrable _ hsm (B := ∑ i, B i)
        (fun s => by simpa only [NNReal.coe_sum] using hbound s)
    have hid (p : Sample X n × X) : sumStat v p =
        sumStat v' p.1 + v (Fin.last n) p.2 := by
      simp only [sumStat, Fin.sum_univ_castSucc, observations,
        Fin.snoc_castSucc, Fin.snoc_last, v']
    change (∫ p : Sample X n × X, ‖sumStat v p‖ ^ 2 ∂(law n μ').prod (μ (Fin.last n))) = _
    rw [integral_prod _ hsqi]
    simp_rw [hid, integral_norm_add_sq (μ (Fin.last n)) (hv (Fin.last n))
      (hb (Fin.last n)) (hcenter (Fin.last n))]
    have hprev : Integrable (fun s => ‖sumStat v' s‖ ^ 2) (law n μ') :=
      bounded_square_integrable _ (sumStat_measurable (fun i => hv i.castSucc))
        (B := ∑ i, B' i) (fun s => by
          simpa only [NNReal.coe_sum] using sumStat_bound (fun i => hb i.castSucc) s)
    rw [integral_add hprev (integrable_const _)]
    rw [ih μ' v' (fun i => hv i.castSucc) B' (fun i => hb i.castSucc)
      (fun i => hcenter i.castSucc)]
    simp only [integral_const, probReal_univ, one_smul, Fin.sum_univ_castSucc, v', μ']

omit [MeasurableSpace X] [InnerProductSpace ℝ E] [CompleteSpace E] in
theorem sumStat_change {n : ℕ} (v : Fin n → X → E) (i : Fin n) (s t : Sample X n)
    (hst : ∀ j, j ≠ i → observations n s j = observations n t j) :
    sumStat v s - sumStat v t = v i (observations n s i) - v i (observations n t i) := by
  classical
  rw [sumStat, sumStat, ← Finset.sum_sub_distrib]
  apply Finset.sum_eq_single i
  · intro j _ hji
    rw [hst j hji, sub_self]
  · simp

omit [MeasurableSpace X] [InnerProductSpace ℝ E] [CompleteSpace E] in
theorem sumNorm_differences {n : ℕ} (v : Fin n → X → E) (c : Fin n → ℝ≥0)
    (hd : ∀ i x y, ‖v i x - v i y‖ ≤ c i) :
    HasDifferences (fun s => ‖sumStat v s‖) c := by
  intro i s t hst
  exact (abs_norm_sub_norm_le _ _).trans (by rw [sumStat_change v i s t hst]; exact hd i _ _)

omit [InnerProductSpace ℝ E] [CompleteSpace E] in
theorem expected_norm_le_sqrt (μ : Measure X) [IsProbabilityMeasure μ]
    {v : X → E} (hv : StronglyMeasurable v) {B : ℝ≥0} (hb : ∀ x, ‖v x‖ ≤ B) :
    (∫ x, ‖v x‖ ∂μ) ≤ Real.sqrt (∫ x, ‖v x‖ ^ 2 ∂μ) := by
  have hdiag (x : X) : ‖(‖v x‖ : ℝ)‖ ^ 2 ≤ (B : ℝ) ^ 2 := by
    simpa only [Real.norm_eq_abs, abs_of_nonneg (norm_nonneg _)] using
      (sq_le_sq₀ (norm_nonneg _) B.coe_nonneg).2 (hb x)
  have hvar := Moments.feature_variance_identity μ hv.norm hdiag
  have hp : 0 ≤ ∫ x, ‖(‖v x‖ : ℝ) - Moments.mean μ (fun x => ‖v x‖)‖ ^ 2 ∂μ :=
    integral_nonneg fun _ => sq_nonneg _
  simp only [Real.norm_eq_abs, sq_abs, Moments.mean] at hvar hp
  apply (Real.le_sqrt (integral_nonneg fun x => norm_nonneg _)
    (integral_nonneg fun x => sq_nonneg _)).2
  linarith

def centeredFamily {n : ℕ} (μ : Fin n → Measure X) (φ : X → E) (w : Fin n → ℝ)
    (i : Fin n) (x : X) : E := w i • (φ x - Moments.mean (μ i) φ)

def familyBound {n : ℕ} (μ : Fin n → Measure X) (φ : X → E) (w : Fin n → ℝ)
    (K : ℝ) (i : Fin n) : ℝ≥0 :=
  NNReal.mk (|w i| * (Real.sqrt K + ‖Moments.mean (μ i) φ‖)) (by positivity)

omit [CompleteSpace E] in
theorem centeredFamily_bound {n : ℕ} (μ : Fin n → Measure X) (φ : X → E)
    (w : Fin n → ℝ) {K : ℝ} (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (i : Fin n) (x : X) :
    ‖centeredFamily μ φ w i x‖ ≤ familyBound μ φ w K i := by
  simp only [centeredFamily, familyBound, NNReal.coe_mk, norm_smul, Real.norm_eq_abs]
  exact mul_le_mul_of_nonneg_left
    ((norm_sub_le _ _).trans (add_le_add (Moments.feature_norm_bound hdiag x) le_rfl))
    (abs_nonneg _)

omit [CompleteSpace E] in
theorem centeredFamily_measurable {n : ℕ} (μ : Fin n → Measure X) (w : Fin n → ℝ)
    {φ : X → E} (hφ : StronglyMeasurable φ) (i : Fin n) :
    StronglyMeasurable (centeredFamily μ φ w i) :=
  (hφ.sub stronglyMeasurable_const).const_smul (w i)

theorem centeredFamily_center {n : ℕ} (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)] (w : Fin n → ℝ) {φ : X → E} {K : ℝ}
    (hφ : StronglyMeasurable φ) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (i : Fin n) :
    ∫ x, centeredFamily μ φ w i x ∂μ i = 0 := by
  simp only [centeredFamily, integral_smul, Moments.centered_integral (μ i) hφ hdiag, smul_zero]

theorem centeredFamily_variance {n : ℕ} (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)] (w : Fin n → ℝ) {φ : X → E} {K : ℝ}
    (hφ : StronglyMeasurable φ) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) (i : Fin n) :
    (∫ x, ‖centeredFamily μ φ w i x‖ ^ 2 ∂μ i) ≤ w i ^ 2 * K := by
  simp only [centeredFamily, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
  rw [integral_const_mul]
  exact mul_le_mul_of_nonneg_left (Moments.feature_variance_upper (μ i) hφ hdiag)
    (sq_nonneg _)

theorem centeredSum_variance (n : ℕ) (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)] (w : Fin n → ℝ) {φ : X → E} {K : ℝ}
    (hφ : StronglyMeasurable φ) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K) :
    (∫ s, ‖sumStat (centeredFamily μ φ w) s‖ ^ 2 ∂law n μ) ≤ K * ∑ i, w i ^ 2 := by
  rw [sumStat_variance n μ _ (centeredFamily_measurable μ w hφ)
    (familyBound μ φ w K) (centeredFamily_bound μ φ w hdiag)
    (centeredFamily_center μ w hφ hdiag)]
  calc
    (∑ i, ∫ x, ‖centeredFamily μ φ w i x‖ ^ 2 ∂μ i) ≤ ∑ i, w i ^ 2 * K :=
      Finset.sum_le_sum fun i _ => centeredFamily_variance μ w hφ hdiag i
    _ = K * ∑ i, w i ^ 2 := by rw [← Finset.sum_mul]; ring

omit [CompleteSpace E] in
theorem centeredFamily_diameter {n : ℕ} (μ : Fin n → Measure X) (w : Fin n → ℝ)
    {φ : X → E} {K : ℝ} (hK : 0 ≤ K) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K)
    (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫) (i : Fin n) (x y : X) :
    ‖centeredFamily μ φ w i x - centeredFamily μ φ w i y‖ ≤ |w i| * Real.sqrt (2 * K) := by
  have hid : centeredFamily μ φ w i x - centeredFamily μ φ w i y =
      w i • (φ x - φ y) := by unfold centeredFamily; module
  rw [hid, norm_smul, Real.norm_eq_abs]
  apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
  exact (Real.le_sqrt (norm_nonneg _) (by positivity)).2
    (Moments.feature_diameter hdiag hkernel x y)

omit [CompleteSpace E] in
/-- The exponential tail behind both confidence radii. -/
theorem weighted_mean_tail (n : ℕ) (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)] (w : Fin n → ℝ) {φ : X → E} {K t : ℝ}
    (hφ : StronglyMeasurable φ) (hK : 0 ≤ K) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K)
    (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫) (ht : 0 ≤ t) :
    (law n μ).real {s | t ≤ ‖sumStat (centeredFamily μ φ w) s‖ -
      ∫ q, ‖sumStat (centeredFamily μ φ w) q‖ ∂law n μ} ≤
        Real.exp (-t ^ 2 / (K * ∑ i, w i ^ 2)) := by
  let : IsProbabilityMeasure (law n μ) := law_isProbability n μ (fun _ => inferInstance)
  let v := centeredFamily μ φ w
  let b := familyBound μ φ w K
  let c : Fin n → ℝ≥0 := fun i =>
    NNReal.mk (|w i| * Real.sqrt (2 * K)) (by positivity)
  have hsm := sumStat_measurable (centeredFamily_measurable μ w hφ)
  have hbound (s : Sample X n) : |‖sumStat v s‖| ≤ ∑ i, (b i : ℝ) := by
    simpa only [abs_of_nonneg (norm_nonneg _)] using
      sumStat_bound (centeredFamily_bound μ φ w hdiag) s
  have hdiff : HasDifferences (fun s => ‖sumStat v s‖) c :=
    sumNorm_differences v c (centeredFamily_diameter μ w hK hdiag hkernel)
  have h := (subgaussian_of_hasDifferences n μ hsm.norm.measurable hbound c hdiff).measure_ge_le ht
  have hcoef : (2 : ℝ) * (↑(∑ i, (c i / 2) ^ 2 : ℝ≥0) : ℝ) =
      K * ∑ i, w i ^ 2 := by
    simp only [NNReal.coe_sum, NNReal.coe_pow, NNReal.coe_div, NNReal.coe_ofNat,
      c, NNReal.coe_mk, div_pow, mul_pow, sq_abs,
      Real.sq_sqrt (by positivity : 0 ≤ 2 * K)]
    rw [Finset.mul_sum, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rwa [hcoef] at h

/-- Weighted Hilbert mean concentration, derived from the sampling law and
the bounded nonnegative kernel, with the sharp constants used in the paper. -/
theorem weighted_mean_concentration (n : ℕ) (μ : Fin n → Measure X)
    [∀ i, IsProbabilityMeasure (μ i)] (w : Fin n → ℝ) {φ : X → E} {K γ : ℝ}
    (hφ : StronglyMeasurable φ) (hK : 0 ≤ K) (hdiag : ∀ x, ‖φ x‖ ^ 2 ≤ K)
    (hkernel : ∀ x y, 0 ≤ ⟪φ x, φ y⟫)
    (hV : 0 < K * ∑ i, w i ^ 2) (hγ : 0 < γ) (hγ₁ : γ < 1) :
    law n μ {s | Real.sqrt (K * ∑ i, w i ^ 2) *
      (1 + Real.sqrt (Real.log (1 / γ))) < ‖sumStat (centeredFamily μ φ w) s‖} ≤
      ENNReal.ofReal γ := by
  let : IsProbabilityMeasure (law n μ) := law_isProbability n μ (fun _ => inferInstance)
  let v := centeredFamily μ φ w
  let b := familyBound μ φ w K
  let c : Fin n → ℝ≥0 := fun i =>
    NNReal.mk (|w i| * Real.sqrt (2 * K)) (by positivity)
  have hsm := sumStat_measurable (centeredFamily_measurable μ w hφ)
  have hb (s : Sample X n) : ‖sumStat v s‖ ≤ ∑ i, (b i : ℝ) :=
    sumStat_bound (centeredFamily_bound μ φ w hdiag) s
  have hbounded (s : Sample X n) : |‖sumStat v s‖| ≤ ∑ i, (b i : ℝ) := by
    simpa only [abs_of_nonneg (norm_nonneg _)] using hb s
  have hdiff : HasDifferences (fun s => ‖sumStat v s‖) c :=
    sumNorm_differences v c (centeredFamily_diameter μ w hK hdiag hkernel)
  have hmg := subgaussian_of_hasDifferences n μ hsm.norm.measurable hbounded c hdiff
  have hcoef : (∑ i, (c i / 2) ^ 2 : ℝ≥0) =
      NNReal.mk ((K * ∑ i, w i ^ 2) / 2) (by positivity) := by
    apply NNReal.eq
    simp only [NNReal.coe_sum, NNReal.coe_pow, NNReal.coe_div, NNReal.coe_ofNat,
      c, NNReal.coe_mk, div_pow, mul_pow, sq_abs, Real.sq_sqrt (by positivity : 0 ≤ 2 * K)]
    rw [Finset.mul_sum, Finset.sum_div]
    apply Finset.sum_congr rfl
    intro i _
    ring
  have hcpos : 0 < (∑ i, (c i / 2) ^ 2 : ℝ≥0) := by
    rw [hcoef]
    exact show (0 : ℝ) < (K * ∑ i, w i ^ 2) / 2 by positivity
  have hprob := confidence_of_subgaussian (law n μ) hmg hcpos hγ hγ₁
  have hmean : (∫ s, ‖sumStat v s‖ ∂law n μ) ≤ Real.sqrt (K * ∑ i, w i ^ 2) := by
    apply (expected_norm_le_sqrt (law n μ) hsm (B := ∑ i, b i)
      (fun s => by simpa only [NNReal.coe_sum] using hb s)).trans
    exact Real.sqrt_le_sqrt (centeredSum_variance n μ w hφ hdiag)
  have hlog : 0 ≤ Real.log (1 / γ) :=
    Real.log_nonneg ((le_div_iff₀ hγ).2 (by linarith))
  have hrad : Real.sqrt (2 * (↑(∑ i, (c i / 2) ^ 2 : ℝ≥0) : ℝ) * Real.log (1 / γ)) =
      Real.sqrt (K * ∑ i, w i ^ 2) * Real.sqrt (Real.log (1 / γ)) := by
    rw [hcoef, NNReal.coe_mk]
    have hid : 2 * ((K * ∑ i, w i ^ 2) / 2) = K * ∑ i, w i ^ 2 := by ring
    rw [hid, Real.sqrt_mul hV.le]
  apply (measure_mono ?_).trans hprob
  intro s hs
  change (∫ q, ‖sumStat v q‖ ∂law n μ) +
    Real.sqrt (2 * (↑(∑ i, (c i / 2) ^ 2 : ℝ≥0) : ℝ) * Real.log (1 / γ)) < ‖sumStat v s‖
  rw [hrad]
  change Real.sqrt (K * ∑ i, w i ^ 2) * (1 + Real.sqrt (Real.log (1 / γ))) <
    ‖sumStat v s‖ at hs
  nlinarith

end Sampling
end PearsonClassification
