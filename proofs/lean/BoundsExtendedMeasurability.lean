module

public import BoundsExtendedStatistical

/-! Continuity of the finite-sample score, including the regularized inverse. -/

@[expose] public section
noncomputable section
namespace PearsonClassification.Statistical
open MeasureTheory InnerProductSpace
open scoped RealInnerProductSpace
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- The inverse in the score is the actual inverse in the operator algebra. -/
theorem regularized_inverse_eq (T : E →L[ℝ] E) (hT : T.IsPositive)
    (reg : ℝ) (hreg : 0 < reg) :
    Ring.inverse (Operators.regularized T reg) =
      (Operators.regularizedEquiv T hT reg hreg).symm.toContinuousLinearMap := by
  rw [← Operators.regularizedEquiv_coe T hT reg hreg,
    ContinuousLinearMap.ringInverse_equiv, ContinuousLinearMap.inverse_equiv]

def vectorScore {n : ℕ} (reg : ℝ) (z : (Fin n → E) × (Fin n → E)) : ℝ :=
  let d := Moments.empiricalMean id z.2 - Moments.empiricalMean id z.1
  ⟪d, (Ring.inverse (Operators.regularized
    (Moments.empiricalSecondMoment id z.1) reg)) d⟫ / 2

/-- Continuous score as a function of the two lists of feature vectors. -/
theorem continuous_featureScore (n : ℕ) (reg : ℝ) (hreg : 0 < reg) :
    Continuous (vectorScore (n := n) (E := E) reg) := by
  let T : ((Fin n → E) × (Fin n → E)) → (E →L[ℝ] E) :=
    fun z => Moments.empiricalSecondMoment id z.1
  let A := fun z => Operators.regularized (T z) reg
  let d : ((Fin n → E) × (Fin n → E)) → E :=
    fun z => Moments.empiricalMean id z.2 - Moments.empiricalMean id z.1
  have hT : Continuous T := by
    dsimp only [T, Moments.empiricalSecondMoment, id_eq]
    apply Continuous.fun_const_smul
    apply continuous_finsetSum
    intro i _
    exact (rankOne ℝ).continuous.comp
      ((continuous_apply i).comp continuous_fst) |>.clm_apply
        ((continuous_apply i).comp continuous_fst)
  have hA : Continuous A := hT.add continuous_const
  have hd : Continuous d := by
    dsimp only [d, Moments.empiricalMean, id_eq]
    fun_prop
  have hinv : Continuous (fun z => Ring.inverse (A z)) := by
    apply continuous_iff_continuousAt.mpr
    intro z
    have hu := Operators.coercive_isUnit (A z) hreg
      (Operators.regularized_coercive (T z)
        (Moments.empiricalSecondMoment_positive id z.1) reg)
    have hi : ContinuousAt Ring.inverse (A z) := by
      simpa only [hu.unit_spec] using NormedRing.inverse_continuousAt hu.unit
    exact hi.comp hA.continuousAt
  have hscore : Continuous (fun z => ⟪d z, (Ring.inverse (A z)) (d z)⟫ / 2) :=
    (hd.inner (hinv.clm_apply hd)).div_const 2
  exact hscore

variable {X : Type*} [MeasurableSpace X]

omit [MeasurableSpace X] in
theorem batchScore_eq_vectorScore {n : ℕ} (φ : X → E) (reg : ℝ) (hreg : 0 < reg)
    (s : BoundedDifferences.Sample X (n + n)) :
    batchScore φ reg hreg s = vectorScore reg
      ((fun i => φ (reference s i)), (fun i => φ (incoming s i))) := by
  unfold batchScore Moments.empiricalScore Operators.quadraticScore vectorScore
  rw [regularized_inverse_eq _ (Moments.empiricalSecondMoment_positive _ _)]
  rfl

/-- Score events are measurable; the probability bounds are ordinary event
probabilities as well as valid outer-measure bounds. -/
theorem batchScore_measurable [TopologicalSpace.SeparableSpace E] {n : ℕ}
    {φ : X → E} (hφ : StronglyMeasurable φ)
    (reg : ℝ) (hreg : 0 < reg) : Measurable (batchScore (n := n) φ reg hreg) := by
  let : MeasurableSpace E := borel E
  have : BorelSpace E := ⟨rfl⟩
  have href : Measurable (fun s : BoundedDifferences.Sample X (n + n) =>
      (fun i => φ (reference s i))) :=
    measurable_pi_iff.mpr fun i => hφ.measurable.comp
      (BoundedDifferences.measurable_observation _ _)
  have hin : Measurable (fun s : BoundedDifferences.Sample X (n + n) =>
      (fun i => φ (incoming s i))) :=
    measurable_pi_iff.mpr fun i => hφ.measurable.comp
      (BoundedDifferences.measurable_observation _ _)
  have hid : batchScore (n := n) φ reg hreg =
      (fun s => vectorScore reg
        ((fun i => φ (reference s i)), (fun i => φ (incoming s i)))) :=
    funext (batchScore_eq_vectorScore φ reg hreg)
  rw [hid]
  exact (continuous_featureScore (E := E) n reg hreg).measurable.comp (href.prodMk hin)

end PearsonClassification.Statistical
