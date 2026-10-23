module

public import BoundsExtendedMoments

/-!
The completed Hilbert tensor square supplies the Hilbert-space representation
of rank-one operators used in the operator-concentration argument. The map
to bounded operators is proved contractive; there is no finite-dimensional
assumption and the identity operator is never asserted Hilbert--Schmidt.
-/

@[expose] public section

noncomputable section

namespace PearsonClassification
namespace HilbertTensor

open InnerProductSpace
open scoped TensorProduct RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def rankOneBilinear : E →ₗ[ℝ] E →ₗ[ℝ] (E →L[ℝ] E) :=
  LinearMap.mk₂ ℝ (fun u v => rankOne ℝ u v)
    (by intro u v w; ext z; simp [rankOne_apply])
    (by intro c u v; ext z; simp [rankOne_apply, smul_smul])
    (by intro u v w; ext z; simp [rankOne_apply])
    (by intro c u v; ext z; simp [rankOne_apply, smul_smul])

def finiteRankOperator : (E ⊗[ℝ] E) →ₗ[ℝ] (E →L[ℝ] E) := TensorProduct.lift rankOneBilinear

@[simp] theorem finiteRankOperator_tmul (u v : E) :
    finiteRankOperator (u ⊗ₜ[ℝ] v) = rankOne ℝ u v := rfl

theorem pairing (t : E ⊗[ℝ] E) (f g : E) :
    ⟪f, finiteRankOperator t g⟫ = ⟪f ⊗ₜ[ℝ] g, t⟫ := by
  induction t using TensorProduct.inductionOn with
  | tmul u v =>
    simp only [finiteRankOperator_tmul, rankOne_apply, inner_smul_right, TensorProduct.inner_tmul]
    rw [real_inner_comm g v]
    ring
  | add s t hs ht =>
    simp only [map_add, add_apply, inner_add_right, hs, ht]

theorem finiteRankOperator_norm_le (t : E ⊗[ℝ] E) : ‖finiteRankOperator t‖ ≤ ‖t‖ := by
  apply ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg t)
  intro g
  have hcs := norm_inner_le_norm (𝕜 := ℝ) (finiteRankOperator t g ⊗ₜ[ℝ] g) t
  rw [← pairing, real_inner_self_eq_norm_sq] at hcs
  simp only [Real.norm_eq_abs, abs_sq, TensorProduct.norm_tmul] at hcs
  by_cases hz : ‖finiteRankOperator t g‖ = 0
  · simp [hz, mul_nonneg (norm_nonneg t) (norm_nonneg g)]
  · have hp : 0 < ‖finiteRankOperator t g‖ := lt_of_le_of_ne (norm_nonneg _) (Ne.symm hz)
    apply (mul_le_mul_iff_left₀ hp).1
    nlinarith [hcs]

abbrev Space (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E] :=
  UniformSpace.Completion (E ⊗[ℝ] E)

def liftRankOne (v : E) : Space E := ((v ⊗ₜ[ℝ] v : E ⊗[ℝ] E) : Space E)

@[simp] theorem norm_liftRankOne (v : E) : ‖liftRankOne v‖ = ‖v‖ ^ 2 := by
  simp [liftRankOne, pow_two]

@[simp] theorem inner_liftRankOne (u v : E) : ⟪liftRankOne u, liftRankOne v⟫ = ⟪u, v⟫ ^ 2 := by
  simp [liftRankOne, pow_two]

theorem rankOne_diameter {K : ℝ} (hK : 0 ≤ K) {u v : E}
    (hu : ‖u‖ ^ 2 ≤ K) (hv : ‖v‖ ^ 2 ≤ K) :
    ‖liftRankOne u - liftRankOne v‖ ^ 2 ≤ 2 * K ^ 2 := by
  rw [norm_sub_sq_real, norm_liftRankOne, norm_liftRankOne, inner_liftRankOne]
  have hu2 : (‖u‖ ^ 2) ^ 2 ≤ K ^ 2 := (sq_le_sq₀ (sq_nonneg _) hK).2 hu
  have hv2 : (‖v‖ ^ 2) ^ 2 ≤ K ^ 2 := (sq_le_sq₀ (sq_nonneg _) hK).2 hv
  nlinarith [sq_nonneg ⟪u, v⟫]

variable [CompleteSpace E]

def toOperator : Space E →L[ℝ] (E →L[ℝ] E) :=
  finiteRankOperator.extendOfNorm UniformSpace.Completion.toComplL.toLinearMap

theorem toOperator_norm_le (t : Space E) : ‖toOperator t‖ ≤ ‖t‖ := by
  have h := LinearMap.norm_extendOfNorm_apply_le
    (f := (finiteRankOperator : (E ⊗[ℝ] E) →ₗ[ℝ] (E →L[ℝ] E)))
    (e := UniformSpace.Completion.toComplL.toLinearMap)
    UniformSpace.Completion.denseRange_coe 1
    (fun s => by simpa using finiteRankOperator_norm_le s) t
  simpa only [toOperator, one_mul] using h

@[simp] theorem toOperator_liftRankOne (v : E) : toOperator (liftRankOne v) = rankOne ℝ v v := by
  have h := LinearMap.extendOfNorm_eq
    (f := (finiteRankOperator : (E ⊗[ℝ] E) →ₗ[ℝ] (E →L[ℝ] E)))
    (e := UniformSpace.Completion.toComplL.toLinearMap)
    UniformSpace.Completion.denseRange_coe
    ⟨1, fun s => by simpa using finiteRankOperator_norm_le s⟩ (v ⊗ₜ[ℝ] v)
  exact h

theorem error_domination (s t : Space E) : ‖toOperator s - toOperator t‖ ≤ ‖s - t‖ := by
  rw [← map_sub]
  exact toOperator_norm_le _

end HilbertTensor
end PearsonClassification
