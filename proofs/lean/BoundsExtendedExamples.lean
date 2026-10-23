module

public import BoundsExtendedFoundations

/-! Exact finite probability models for the two appendix counterexamples. -/

@[expose] public section
noncomputable section
namespace PearsonClassification.Examples
open MeasureTheory InnerProductSpace
open scoped RealInnerProductSpace ENNReal

/-- Bernoulli probability, with `true` having mass `p`. -/
def twoPoint (p : ℝ) : Measure Bool :=
  ENNReal.ofReal (1 - p) • Measure.dirac false + ENNReal.ofReal p • Measure.dirac true

theorem twoPoint_probability {p : ℝ} (hp : 0 ≤ p) (hp₁ : p ≤ 1) :
    IsProbabilityMeasure (twoPoint p) := by
  constructor
  simp only [twoPoint, Measure.add_apply, Measure.smul_apply, Measure.dirac_apply_of_mem,
    Set.mem_univ, smul_eq_mul, mul_one]
  rw [← ENNReal.ofReal_add (by linarith : 0 ≤ 1 - p) hp]
  norm_num

theorem twoPoint_integral {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [CompleteSpace E] {p : ℝ} (hp : 0 ≤ p) (hp₁ : p ≤ 1) (f : Bool → E) :
    ∫ x, f x ∂twoPoint p = (1 - p) • f false + p • f true := by
  unfold twoPoint
  rw [integral_add_measure (Integrable.of_finite.smul_measure ENNReal.ofReal_ne_top)
    (Integrable.of_finite.smul_measure ENNReal.ofReal_ne_top),
    integral_smul_measure, integral_smul_measure, integral_dirac, integral_dirac,
    ENNReal.toReal_ofReal (by linarith : 0 ≤ 1 - p), ENNReal.toReal_ofReal hp]

section Orthonormal
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- Orthonormal feature vectors realize the two-point indicator kernel. -/
def orthogonalFeature (a b : E) (x : Bool) : E := if x then a else b

omit [CompleteSpace E] in
theorem orthogonalFeature_inner (a b : E) (ha : ‖a‖ ^ 2 = 1) (hb : ‖b‖ ^ 2 = 1)
    (hab : ⟪a, b⟫ = 0) (x y : Bool) :
    ⟪orthogonalFeature a b x, orthogonalFeature a b y⟫ = if x = y then 1 else 0 := by
  have hba : ⟪b, a⟫ = 0 := by rw [real_inner_comm a b, hab]
  cases x <;> cases y <;> simp [orthogonalFeature, ha, hb, hab, hba]

theorem orthogonal_mean {p : ℝ} (hp : 0 ≤ p) (hp₁ : p ≤ 1) (a b : E) :
    Moments.mean (twoPoint p) (orthogonalFeature a b) = (1 - p) • b + p • a := by
  exact twoPoint_integral hp hp₁ _

omit [CompleteSpace E] in
theorem orthogonal_point_error (p : ℝ) (a b : E)
    (ha : ‖a‖ ^ 2 = 1) (hb : ‖b‖ ^ 2 = 1) (hab : ⟪a, b⟫ = 0) :
    ‖a - ((1 - p) • b + p • a)‖ ^ 2 = 2 * (1 - p) ^ 2 := by
  rw [norm_sub_sq_real, norm_add_sq_real]
  simp only [norm_smul, Real.norm_eq_abs, mul_pow, sq_abs, ha, hb,
    inner_smul_left, inner_smul_right, inner_add_right, real_inner_self_eq_norm_sq,
    conj_trivial, hab, mul_zero, zero_add, mul_one]
  have hba : ⟪b, a⟫ = 0 := by rw [real_inner_comm a b, hab]
  rw [hba]
  ring

theorem orthogonal_variance {p : ℝ} (hp : 0 ≤ p) (hp₁ : p ≤ 1) (a b : E)
    (ha : ‖a‖ ^ 2 = 1) (hb : ‖b‖ ^ 2 = 1) (hab : ⟪a, b⟫ = 0) :
    (∫ x, ‖orthogonalFeature a b x -
      Moments.mean (twoPoint p) (orthogonalFeature a b)‖ ^ 2 ∂twoPoint p) =
      2 * p * (1 - p) := by
  rw [twoPoint_integral hp hp₁, orthogonal_mean hp hp₁]
  change (1 - p) * ‖b - ((1 - p) • b + p • a)‖ ^ 2 +
    p * ‖a - ((1 - p) • b + p • a)‖ ^ 2 = _
  rw [orthogonal_point_error p a b ha hb hab]
  have hba : ⟪b, a⟫ = 0 := by rw [real_inner_comm a b, hab]
  have h := orthogonal_point_error (1 - p) b a hb ha hba
  have hid : (1 - (1 - p)) • a + (1 - p) • b = (1 - p) • b + p • a := by module
  rw [hid] at h
  rw [h]
  ring

theorem orthogonal_variance_upper {p : ℝ} (hp : 0 ≤ p) (hp₁ : p ≤ 1) :
    0 ≤ 2 * p * (1 - p) ∧ 2 * p * (1 - p) ≤ (1 : ℝ) / 2 := by
  constructor
  · positivity
  · nlinarith [sq_nonneg (p - 1 / 2)]

theorem orthogonal_example_numbers :
    2 * (1 - (1 : ℝ) / 100) ^ 2 = 19602 / 10000 ∧
    1 < 2 * (1 - (1 : ℝ) / 100) ^ 2 ∧
    2 * ((1 : ℝ) / 100) * (1 - (1 : ℝ) / 100) = 198 / 10000 ∧
    2 * ((1 : ℝ) / 100) * (1 - (1 : ℝ) / 100) < 1 := by norm_num

theorem orthonormal_feature_exists : ∃ a b : EuclideanSpace ℝ Bool,
    ‖a‖ ^ 2 = 1 ∧ ‖b‖ ^ 2 = 1 ∧ ⟪a, b⟫ = 0 := by
  refine ⟨EuclideanSpace.single true 1, EuclideanSpace.single false 1, ?_⟩
  norm_num [EuclideanSpace.inner_single_left]

end Orthonormal

def bit (x : Bool) : ℝ := if x then 1 else 0

def scalarScore (x y : Bool) : ℝ := (bit y - bit x) ^ 2 / (2 * (1 + bit x))

/-- The displayed scalar formula is the actual empirical inverse score. -/
theorem scalarScore_eq_empiricalScore (x y : Bool) :
    Moments.empiricalScore bit (fun _ : Fin 1 => x) (fun _ : Fin 1 => y) 1
      (by norm_num) = scalarScore x y := by
  let A := Operators.regularizedEquiv
    (Moments.empiricalSecondMoment bit (fun _ : Fin 1 => x))
    (Moments.empiricalSecondMoment_positive bit _) 1 (by norm_num)
  have hA (f : ℝ) : A f = (1 + bit x) * f := by
    change A.toContinuousLinearMap f = _
    rw [Operators.regularizedEquiv_coe]
    cases x <;> simp [Operators.regularized, Moments.empiricalSecondMoment,
      bit]
    ring
  have hinv : A.symm (bit y - bit x) = (bit y - bit x) / (1 + bit x) := by
    apply A.injective
    rw [ContinuousLinearEquiv.apply_symm_apply, hA]
    cases x <;> simp [bit]
    ring
  change Operators.quadraticScore A _ = _
  simp only [Moments.empiricalMean, Nat.cast_one, inv_one, Fin.sum_univ_one, one_smul]
  rw [Operators.quadraticScore, hinv]
  rw [Real.inner_apply]
  cases x <;> cases y <;> norm_num [scalarScore, bit]

theorem twoPoint_prod_integral {p : ℝ} (hp : 0 ≤ p) (hp₁ : p ≤ 1)
    (f : Bool × Bool → ℝ) :
    ∫ z, f z ∂(twoPoint p).prod (twoPoint p) =
      (1 - p) * ((1 - p) * f (false, false) + p * f (false, true)) +
      p * ((1 - p) * f (true, false) + p * f (true, true)) := by
  let : IsProbabilityMeasure (twoPoint p) := twoPoint_probability hp hp₁
  rw [integral_prod _ Integrable.of_finite, twoPoint_integral hp hp₁]
  simp_rw [twoPoint_integral hp hp₁, smul_eq_mul]

theorem coupled_score_values : scalarScore false true = 1 / 2 ∧
    scalarScore true false = 1 / 4 ∧ scalarScore false false = 0 ∧
    scalarScore true true = 0 := by norm_num [scalarScore, bit]

theorem coupled_score_expectation :
    ∫ z, scalarScore z.1 z.2 ∂(twoPoint (1 / 5)).prod (twoPoint (1 / 5)) =
      (12 : ℝ) / 100 := by
  rw [twoPoint_prod_integral (by norm_num) (by norm_num)]
  norm_num [scalarScore, bit]

theorem coupled_failed_factorization :
    (1 / 2 : ℝ) * (∫ z, (bit z.2 - bit z.1) ^ 2
      ∂(twoPoint (1 / 5)).prod (twoPoint (1 / 5))) *
      (∫ x, 1 / (1 + bit x) ∂twoPoint (1 / 5)) = 144 / 1000 ∧
    (∫ z, scalarScore z.1 z.2 ∂(twoPoint (1 / 5)).prod (twoPoint (1 / 5))) ≠
      (1 / 2 : ℝ) * (∫ z, (bit z.2 - bit z.1) ^ 2
        ∂(twoPoint (1 / 5)).prod (twoPoint (1 / 5))) *
        (∫ x, 1 / (1 + bit x) ∂twoPoint (1 / 5)) := by
  rw [twoPoint_prod_integral (by norm_num) (by norm_num),
    twoPoint_integral (by norm_num) (by norm_num), coupled_score_expectation]
  norm_num [bit]

theorem coupled_off_diagonal_mass :
    (twoPoint (1 / 5)).prod (twoPoint (1 / 5)) ({false} ×ˢ {true}) =
      ENNReal.ofReal (16 / 100) ∧
    (twoPoint (1 / 5)).prod (twoPoint (1 / 5)) ({true} ×ˢ {false}) =
      ENNReal.ofReal (16 / 100) := by
  simp only [Measure.prod_prod]
  norm_num [twoPoint, Measure.add_apply, Measure.smul_apply, Measure.dirac_apply]
  constructor <;> rw [← ENNReal.ofReal_mul (by positivity)] <;> norm_num

/-- The normalized Gaussian has the pointwise positivity and diagonal bound
required by the bounds. Positive definiteness and characteristicness are
separate kernel-specific properties. -/
def gaussianKernel {G : Type*} [NormedAddCommGroup G] (σ : ℝ) (x y : G) : ℝ :=
  Real.exp (-‖x - y‖ ^ 2 / (2 * σ ^ 2))

theorem gaussianKernel_bounds {G : Type*} [NormedAddCommGroup G]
    (σ : ℝ) (x y : G) : 0 < gaussianKernel σ x y ∧ gaussianKernel σ x y ≤ 1 := by
  refine ⟨Real.exp_pos _, Real.exp_le_one_iff.mpr ?_⟩
  exact div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (sq_nonneg _)) (by positivity)

theorem gaussianKernel_diagonal {G : Type*} [NormedAddCommGroup G]
    (σ : ℝ) (x : G) : gaussianKernel σ x x = 1 := by simp [gaussianKernel]

end PearsonClassification.Examples
