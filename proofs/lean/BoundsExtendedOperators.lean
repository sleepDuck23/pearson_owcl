module

public import Mathlib

/-!
Actual positive operators behind the Pearson score. These results use real
Hilbert spaces without a finite-dimensionality or commutativity assumption.
The inverse comparison is proved by completing the square, avoiding any
unproved identification of an arbitrary map with an inverse square root.
-/

@[expose] public section

noncomputable section

namespace PearsonClassification
namespace Operators

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The quadratic score from equations `eq:D` and `eq:Dhat`. -/
def quadraticScore (A : E ≃L[ℝ] E) (v : E) : ℝ :=
  ⟪v, A.symm v⟫ / 2

/-- The variational objective in Appendix `app:variational`. -/
def objective (A : E →L[ℝ] E) (v f : E) : ℝ :=
  ⟪f, v⟫ - ⟪f, A f⟫ / 2

lemma inverse_positive (A : E ≃L[ℝ] E)
    (hA : A.toContinuousLinearMap.IsPositive) :
    A.symm.toContinuousLinearMap.IsPositive := by
  exact (LinearEquiv.isPositive_symm_iff).2 hA

lemma quadraticScore_nonneg (A : E ≃L[ℝ] E)
    (hA : A.toContinuousLinearMap.IsPositive) (v : E) :
    0 ≤ quadraticScore A v := by
  exact div_nonneg ((inverse_positive A hA).inner_nonneg_right v) (by norm_num)

theorem completing_square (A : E ≃L[ℝ] E)
    (hA : A.toContinuousLinearMap.IsPositive) (v f : E) :
    objective A.toContinuousLinearMap v f = quadraticScore A v -
      ⟪f - A.symm v, A (f - A.symm v)⟫ / 2 := by
  have hsym := hA.inner_left_eq_inner_right f (A.symm v)
  simp only [ContinuousLinearEquiv.coe_coe, ContinuousLinearEquiv.apply_symm_apply] at hsym
  simp only [objective, quadraticScore, map_sub, ContinuousLinearEquiv.apply_symm_apply,
    inner_sub_left, inner_sub_right, ContinuousLinearEquiv.coe_coe]
  rw [← hsym, real_inner_comm (A.symm v) (A f), real_inner_comm (A.symm v) v]
  ring

theorem objective_le_score (A : E ≃L[ℝ] E)
    (hA : A.toContinuousLinearMap.IsPositive) (v f : E) :
    objective A.toContinuousLinearMap v f ≤ quadraticScore A v := by
  rw [completing_square A hA]
  have := hA.inner_nonneg_right (f - A.symm v)
  change 0 ≤ ⟪f - A.symm v, A (f - A.symm v)⟫ at this
  linarith

theorem objective_at_inverse (A : E ≃L[ℝ] E) (v : E) :
    objective A.toContinuousLinearMap v (A.symm v) = quadraticScore A v := by
  simp only [objective, quadraticScore, ContinuousLinearEquiv.coe_coe,
    ContinuousLinearEquiv.apply_symm_apply, real_inner_comm (A.symm v) v]
  ring

/-- Coercivity makes the variational maximizer unique. -/
theorem objective_eq_score_iff (A : E ≃L[ℝ] E)
    (hA : A.toContinuousLinearMap.IsPositive) {reg : ℝ} (hreg : 0 < reg)
    (hcoercive : ∀ f, reg * ‖f‖ ^ 2 ≤ ⟪f, A f⟫) (v f : E) :
    objective A.toContinuousLinearMap v f = quadraticScore A v ↔ f = A.symm v := by
  constructor
  · intro h
    rw [completing_square A hA] at h
    have hc := hcoercive (f - A.symm v)
    have hn : ‖f - A.symm v‖ ^ 2 = 0 := by nlinarith [sq_nonneg ‖f - A.symm v‖]
    exact sub_eq_zero.mp (norm_eq_zero.mp (sq_eq_zero_iff.mp hn))
  · rintro rfl
    exact objective_at_inverse A v

/-- The score is an attained maximum, and hence the variational supremum. -/
theorem variational_maximum (A : E ≃L[ℝ] E)
    (hA : A.toContinuousLinearMap.IsPositive) (v : E) :
    IsGreatest (Set.range (objective A.toContinuousLinearMap v)) (quadraticScore A v) := by
  refine ⟨⟨A.symm v, objective_at_inverse A v⟩, ?_⟩
  rintro _ ⟨f, rfl⟩
  exact objective_le_score A hA v f

theorem variational_supremum (A : E ≃L[ℝ] E)
    (hA : A.toContinuousLinearMap.IsPositive) (v : E) :
    sSup (Set.range (objective A.toContinuousLinearMap v)) = quadraticScore A v :=
  (variational_maximum A hA v).csSup_eq

/-- Inverse-order comparison in quadratic form, valid for noncommuting operators. -/
theorem inverse_comparison (A B : E ≃L[ℝ] E)
    (_hA : A.toContinuousLinearMap.IsPositive)
    (hB : B.toContinuousLinearMap.IsPositive)
    {c : ℝ} (hc : 0 < c)
    (horder : ∀ f, ⟪f, B f⟫ ≤ c * ⟪f, A f⟫) (v : E) :
    quadraticScore A v ≤ c * quadraticScore B v := by
  have htest := objective_le_score B hB v (c⁻¹ • A.symm v)
  have hbound := horder (c⁻¹ • A.symm v)
  simp only [map_smul, inner_smul_left, inner_smul_right, conj_trivial,
    ContinuousLinearEquiv.apply_symm_apply] at hbound
  simp only [objective, quadraticScore, ContinuousLinearEquiv.coe_coe,
    map_smul, inner_smul_left, inner_smul_right, conj_trivial] at htest ⊢
  rw [← real_inner_comm (A.symm v) v] at htest hbound
  have hc0 : c ≠ 0 := ne_of_gt hc
  have heq : c * c⁻¹ = 1 := mul_inv_cancel₀ hc0
  have hscaled := mul_le_mul_of_nonneg_left htest hc.le
  have hscaled_bound := mul_le_mul_of_nonneg_left hbound hc.le
  field_simp at hscaled hscaled_bound ⊢
  nlinarith

theorem inverse_norm_bound (A : E ≃L[ℝ] E) {reg : ℝ} (hreg : 0 < reg)
    (hcoercive : ∀ f, reg * ‖f‖ ^ 2 ≤ ⟪f, A f⟫) (v : E) :
    ‖A.symm v‖ ≤ ‖v‖ / reg := by
  have h := hcoercive (A.symm v)
  rw [ContinuousLinearEquiv.apply_symm_apply] at h
  have hcs := real_inner_le_norm (A.symm v) v
  apply (le_div_iff₀ hreg).2
  by_cases hz : ‖A.symm v‖ = 0
  · simp [hz]
  · have hp : 0 < ‖A.symm v‖ := lt_of_le_of_ne (norm_nonneg _) (Ne.symm hz)
    nlinarith

theorem quadraticScore_upper (A : E ≃L[ℝ] E) {reg : ℝ} (hreg : 0 < reg)
    (hcoercive : ∀ f, reg * ‖f‖ ^ 2 ≤ ⟪f, A f⟫) (v : E) :
    quadraticScore A v ≤ ‖v‖ ^ 2 / (2 * reg) := by
  have hn := inverse_norm_bound A hreg hcoercive v
  have hcs := real_inner_le_norm v (A.symm v)
  have hp := mul_le_mul_of_nonneg_left hn (norm_nonneg v)
  unfold quadraticScore
  apply (le_div_iff₀ (show 0 < 2 * reg by positivity)).2
  have hm : ⟪v, A.symm v⟫ * reg ≤ ‖v‖ ^ 2 := by
    apply (le_div_iff₀ hreg).1
    calc
      ⟪v, A.symm v⟫ ≤ ‖v‖ * (‖v‖ / reg) := le_trans hcs hp
      _ = ‖v‖ ^ 2 / reg := by ring
  nlinarith

/-- The perturbation bound implies `B ≤ (1 + e / reg) A`. -/
theorem perturbation_order (A B : E ≃L[ℝ] E) {reg e : ℝ}
    (hreg : 0 < reg) (he : 0 ≤ e)
    (hcoercive : ∀ f, reg * ‖f‖ ^ 2 ≤ ⟪f, A f⟫)
    (herror : ‖B.toContinuousLinearMap - A.toContinuousLinearMap‖ ≤ e) (f : E) :
    ⟪f, B f⟫ ≤ (1 + e / reg) * ⟪f, A f⟫ := by
  have hop := (B.toContinuousLinearMap - A.toContinuousLinearMap).le_opNorm f
  have hcs := real_inner_le_norm f ((B.toContinuousLinearMap - A.toContinuousLinearMap) f)
  have hn := mul_le_mul_of_nonneg_right herror (norm_nonneg f)
  have hh := mul_le_mul_of_nonneg_left (le_trans hop hn) (norm_nonneg f)
  have hc := mul_le_mul_of_nonneg_left (hcoercive f) (div_nonneg he hreg.le)
  have hcancel : e / reg * reg = e := div_mul_cancel₀ e (ne_of_gt hreg)
  simp only [sub_apply, ContinuousLinearEquiv.coe_coe] at hh
  simp only [sub_apply, ContinuousLinearEquiv.coe_coe, inner_sub_right] at hcs
  rw [← mul_assoc, hcancel] at hc
  nlinarith

/-! ## Existence of the regularized inverse -/

theorem coercive_isUnit [CompleteSpace E] (A : E →L[ℝ] E)
    {reg : ℝ} (hreg : 0 < reg)
    (hcoercive : ∀ f, reg * ‖f‖ ^ 2 ≤ ⟪f, A f⟫) : IsUnit A := by
  apply A.isUnit_of_forall_le_norm_inner_map (c := ⟨reg, hreg.le⟩) hreg
  intro f
  change ‖f‖ ^ 2 * reg ≤ |⟪A f, f⟫|
  have h := hcoercive f
  rw [← real_inner_comm f (A f)] at h
  exact (by nlinarith : ‖f‖ ^ 2 * reg ≤ ⟪A f, f⟫).trans (le_abs_self _)

def regularized (T : E →L[ℝ] E) (reg : ℝ) : E →L[ℝ] E :=
  T + reg • ContinuousLinearMap.id ℝ E

theorem regularized_positive (T : E →L[ℝ] E) (hT : T.IsPositive)
    {reg : ℝ} (hreg : 0 ≤ reg) : (regularized T reg).IsPositive :=
  hT.add (ContinuousLinearMap.isPositive_id.smul_of_nonneg hreg)

theorem regularized_coercive (T : E →L[ℝ] E) (hT : T.IsPositive)
    (reg : ℝ) (f : E) : reg * ‖f‖ ^ 2 ≤ ⟪f, regularized T reg f⟫ := by
  have h := hT.inner_nonneg_right f
  simp only [regularized, add_apply, smul_apply, ContinuousLinearMap.id_apply,
    inner_add_right, inner_smul_right, real_inner_self_eq_norm_sq]
  linarith

def regularizedEquiv [CompleteSpace E] (T : E →L[ℝ] E) (hT : T.IsPositive)
    (reg : ℝ) (hreg : 0 < reg) : E ≃L[ℝ] E :=
  ContinuousLinearEquiv.ofUnit
    (coercive_isUnit (regularized T reg) hreg (regularized_coercive T hT reg)).unit

@[simp] theorem regularizedEquiv_coe [CompleteSpace E] (T : E →L[ℝ] E)
    (hT : T.IsPositive) (reg : ℝ) (hreg : 0 < reg) :
    (regularizedEquiv T hT reg hreg).toContinuousLinearMap = regularized T reg := by
  exact (coercive_isUnit (regularized T reg) hreg (regularized_coercive T hT reg)).unit_spec

/-! ## Weighted norm, with the actual inverse operator -/

def weightedNorm (A : E ≃L[ℝ] E) (v : E) : ℝ := Real.sqrt ⟪v, A.symm v⟫

lemma weightedNorm_nonneg (A : E ≃L[ℝ] E) (v : E) : 0 ≤ weightedNorm A v :=
  Real.sqrt_nonneg _

lemma weightedNorm_sq (A : E ≃L[ℝ] E)
    (hA : A.toContinuousLinearMap.IsPositive) (v : E) :
    weightedNorm A v ^ 2 = 2 * quadraticScore A v := by
  have hpos : 0 ≤ ⟪v, A.symm v⟫ := (inverse_positive A hA).inner_nonneg_right v
  rw [weightedNorm, Real.sq_sqrt hpos]
  unfold quadraticScore
  ring

theorem weightedNorm_triangle (A : E ≃L[ℝ] E)
    (hA : A.toContinuousLinearMap.IsPositive) (u v : E) :
    weightedNorm A (u + v) ≤ weightedNorm A u + weightedNorm A v := by
  let core : PreInnerProductSpace.Core ℝ E := {
    inner := fun x y => ⟪x, A.symm y⟫
    conj_inner_symm := fun x y => by
      simp only [conj_trivial]
      calc
        ⟪y, A.symm x⟫ = ⟪A.symm y, x⟫ :=
          ((inverse_positive A hA).inner_left_eq_inner_right y x).symm
        _ = ⟪x, A.symm y⟫ := real_inner_comm x (A.symm y)
    re_inner_nonneg := fun x => (inverse_positive A hA).inner_nonneg_right x
    add_left := fun x y z => inner_add_left x y (A.symm z)
    smul_left := fun x y r => by simp [inner_smul_left]
  }
  have h := @norm_add_le E
    (@InnerProductSpace.Core.toSeminormedAddCommGroup ℝ E _ _ _ core).toSeminormedAddGroup u v
  exact h

theorem weightedNorm_upper (A : E ≃L[ℝ] E)
    (hA : A.toContinuousLinearMap.IsPositive) {reg : ℝ} (hreg : 0 < reg)
    (hcoercive : ∀ f, reg * ‖f‖ ^ 2 ≤ ⟪f, A f⟫) (v : E) :
    weightedNorm A v ≤ ‖v‖ / Real.sqrt reg := by
  have h := quadraticScore_upper A hreg hcoercive v
  have hsq := weightedNorm_sq A hA v
  have hroot := Real.sq_sqrt hreg.le
  have hdiv : (‖v‖ / Real.sqrt reg) ^ 2 = ‖v‖ ^ 2 / reg := by
    rw [div_pow, hroot]
  apply (Real.sqrt_le_iff).2
  refine ⟨by positivity, ?_⟩
  rw [hdiv]
  unfold quadraticScore at h
  have hmul := (le_div_iff₀ (show 0 < 2 * reg by positivity)).1 h
  apply (le_div_iff₀ hreg).2
  nlinarith

theorem quadraticScore_smul (A : E ≃L[ℝ] E) (v : E) (t : ℝ) :
    quadraticScore A (t • v) = t ^ 2 * quadraticScore A v := by
  simp only [quadraticScore, map_smul, inner_smul_left, inner_smul_right, conj_trivial]
  ring

theorem quadraticScore_eq_zero_iff (A : E ≃L[ℝ] E) {reg : ℝ}
    (hreg : 0 < reg) (hcoercive : ∀ f, reg * ‖f‖ ^ 2 ≤ ⟪f, A f⟫) (v : E) :
    quadraticScore A v = 0 ↔ v = 0 := by
  constructor
  · intro hz
    have h := hcoercive (A.symm v)
    rw [ContinuousLinearEquiv.apply_symm_apply, real_inner_comm v (A.symm v)] at h
    have hq : ⟪v, A.symm v⟫ = 0 := by unfold quadraticScore at hz; linarith
    have hm : reg * ‖A.symm v‖ ^ 2 = 0 :=
      le_antisymm (by linarith) (mul_nonneg hreg.le (sq_nonneg _))
    have hn : ‖A.symm v‖ = 0 :=
      (sq_eq_zero_iff).1 ((mul_eq_zero.1 hm).resolve_left (ne_of_gt hreg))
    have hi : A.symm v = 0 := norm_eq_zero.1 hn
    have := congrArg A hi
    simpa only [ContinuousLinearEquiv.apply_symm_apply, map_zero] using this
  · rintro rfl
    simp [quadraticScore]

theorem quadraticScore_lower (A : E ≃L[ℝ] E)
    (hA : A.toContinuousLinearMap.IsPositive) {C : ℝ} (hC : 0 < C)
    (hupper : ∀ f, ⟪f, A f⟫ ≤ C * ‖f‖ ^ 2) (v : E) :
    ‖v‖ ^ 2 / (2 * C) ≤ quadraticScore A v := by
  have ht := objective_le_score A hA v (C⁻¹ • v)
  simp only [objective, ContinuousLinearEquiv.coe_coe, map_smul,
    inner_smul_left, inner_smul_right, conj_trivial, real_inner_self_eq_norm_sq] at ht
  have hu := mul_le_mul_of_nonneg_left (hupper v) (sq_nonneg C⁻¹)
  have hC0 := ne_of_gt hC
  have heq : C * C⁻¹ = 1 := mul_inv_cancel₀ hC0
  apply (div_le_iff₀ (show 0 < 2 * C by positivity)).2
  have ht' := mul_le_mul_of_nonneg_left ht hC.le
  have hu' := mul_le_mul_of_nonneg_left hu hC.le
  field_simp at ht' hu' ⊢
  nlinarith

end Operators
end PearsonClassification
