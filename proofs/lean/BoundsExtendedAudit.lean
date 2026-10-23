module

import BoundsExtended
import Lean.Util.CollectAxioms

/-!
Build-time audit of every declaration in the formalization's namespace.
Only Lean's standard logical axioms are allowed. In particular, a proof
using `sorry`, `admit`, or a new axiom fails the build.
This checks proof dependencies; it does not establish semantic coverage of
the TeX manuscript. That correspondence is documented in `COVERAGE.md`.
-/

open Lean in
run_cmd do
  let env ← getEnv
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut declarations : Nat := 0
  for (name, _) in env.constants.toList do
    if (`PearsonClassification).isPrefixOf name then
      declarations := declarations + 1
      for axiomName in ← collectAxioms name do
        unless allowed.contains axiomName do
          throwError "Unapproved axiom {axiomName} in {name}"
  for required in #[``PearsonClassification.Operators.inverse_comparison,
      ``PearsonClassification.nonmerging_quadratic_lower_bound,
      ``PearsonClassification.explicit_calibration_iff,
      ``PearsonClassification.bank_known_calibrated,
      ``PearsonClassification.BoundedDifferences.mcdiarmid_tail,
      ``PearsonClassification.Statistical.mean_concentration,
      ``PearsonClassification.Statistical.operator_concentration,
      ``PearsonClassification.Statistical.null_error,
      ``PearsonClassification.Statistical.alternative_error,
      ``PearsonClassification.Statistical.static_threshold,
      ``PearsonClassification.Statistical.bank_known,
      ``PearsonClassification.Statistical.bank_unknown,
      ``PearsonClassification.Statistical.batchScore_measurable,
      ``PearsonClassification.Statistical.mean_tail,
      ``PearsonClassification.Statistical.operator_tensor_tail,
      ``PearsonClassification.BoundedDifferences.independent_samples_law,
      ``PearsonClassification.Moments.empiricalScore_unique_maximizer,
      ``PearsonClassification.Statistical.finite_family_separation,
      ``PearsonClassification.Examples.coupled_failed_factorization,
      ``PearsonClassification.meanRadius_isBigO] do
    unless env.contains required do
      throwError "Missing required theorem {required}; the audit imported the wrong target"
  logInfo m!"Pearson audit passed: {declarations} declarations; only standard Lean axioms."
