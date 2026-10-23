# Pearson Lean proofs

This Lake project supplies Lean 4.35.0 and Mathlib for the proofs in
`../proofs/lean/`.

From the repository root, build all proof modules and run the axiom audit:

```sh
cd pearson_proof
lake build
```

An exit code of zero means every included module checked successfully.
The audit rejects proof placeholders and axioms other than Lean's standard
`propext`, `Classical.choice`, and `Quot.sound`.

To rerun the audit explicitly after building:

```sh
lake env lean ../proofs/lean/BoundsExtendedAudit.lean
```

`BoundsExtended.lean` is now an entry point importing the separate moment,
operator, tensor, scalar, calibration, and probability modules. After editing
any of those files, use `lake build` to rebuild their compiled imports.

The main concentration and classification bounds now follow from the actual
independent sampling law and bounded nonnegative feature assumptions. The
final theorems are `Statistical.null_error`, `alternative_error`,
`static_threshold`, `bank_known`, and `bank_unknown`. Concentration is proved,
rather than supplied as a premise to these theorems. Score-event measurability,
variational maximizers, calibration, rates, and appendix numerical examples
are also checked.

**This is not a certification of every background assertion in the TeX.**
Gaussian positive definiteness/characteristicness, a literal identification
with the basis-sum Hilbert–Schmidt definition, and the general unbounded
finite-sum variance statement remain outside the coverage. See
[the coverage report](../proofs/lean/COVERAGE.md) for the exact correspondence
and qualifications. Mathematical assumptions such as positive separation
still need justification for any concrete application.

Project warnings are errors, and the default build audits all exposed proof
declarations for unexpected axioms. Always rebuild after editing imported
modules; running Lean on the entry point alone can load older compiled imports.

For a fresh checkout with Lean/Elan installed, first download the dependencies
and their compiled cache from this directory:

```sh
lake update
lake exe cache get
```

The default build includes both the original `PearsonProof` scaffold and the
`BoundsExtended` library, including `BoundsExtendedAudit`.
