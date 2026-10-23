# Formal verification coverage

The target manuscript is [`../bounds_extended.tex`](../bounds_extended.tex).
The entry point is [`BoundsExtended.lean`](BoundsExtended.lean).

**The main concentration, classification, threshold, sample-size, and fixed-bank
results are proved from the kernel/feature and independent-sampling assumptions.**
The final statistical theorems do not assume concentration inequalities,
inverse comparisons, or unnamed good events as additional premises.

**This is not a literal certification of every background assertion in the
TeX document.** The kernel-specific Gaussian properties and the identification
with the basis-sum definition of Hilbert–Schmidt operators remain outside the
formalization; see “Remaining scope” below. The mathematical correspondence
between prose and Lean statements is reviewed here, rather than decided by Lean.

## Reproduce the checks

From the repository root:

```sh
cd pearson_proof
lake build
lake env lean -DwarningAsError=true ../proofs/lean/BoundsExtendedAudit.lean
```

Lean and Mathlib are pinned to v4.35.0. The default build checks all proof
modules and runs the axiom audit. Project warnings are treated as errors.
The audit inspects the transitive axiom dependencies of every exposed
`PearsonClassification` declaration and permits only Lean's standard
`propext`, `Classical.choice`, and `Quot.sound`. A used `sorry`, `admit`, or
custom axiom fails the audit. It also requires the principal theorem names
to be present, preventing a successful audit of an incomplete entry point.

Run `lake build` after edits: running Lean on the entry point alone can load
previously compiled imports. A silent exit with code zero checks the file
and the compiled imports actually loaded; it does not compare them with TeX.

## Final theorems and their assumptions

* `Statistical.mean_concentration` and `operator_concentration` prove the
  two bad-event bounds on the same joint independent batch law.
  `mean_high_probability` and `operator_high_probability` state the
  corresponding good-event probabilities.
* `Statistical.null_error`, `alternative_error`, `null_high_probability`,
  and `alternative_high_probability` apply to the actual regularized
  empirical Pearson score. They establish the manuscript's pairwise bounds.
* `Statistical.static_threshold` packages the two error guarantees for one
  threshold, using a measurable feature map into a separable Hilbert space.
* `Statistical.bank_known` and `bank_unknown` prove the fixed-bank
  classification guarantees from independent observations and their marginal
  distributions. They use actual empirical scores and `α/M`, and do not
  assume independence of the different scores.

As in the manuscript, the assumptions still include a bounded nonnegative
kernel (expressed through feature inner products), a positive sample size,
positive regularization, confidence in `(0,1)`, and the stated population
separation and threshold conditions. Lean proves the implications; it does
not establish a separation assumption for arbitrary distributions or data.

The sampling space is a recursive finite product, not a finite approximation
to the Hilbert space. `BoundedDifferences.observations_map` identifies its
law with `Measure.pi`; `observation_map` and `observations_independent`
prove the marginal and independence properties.
`independent_samples_law` shows that any independent measurable observations
with the prescribed marginals have this encoded sample law. Thus the model
does not impose a special discrete distribution or an extra restriction on
independent sampling. Shared reference observations are used for both
estimators throughout. The finite-bank theorems also allow shared incoming
observations between comparisons.

## Correspondence to the main text

Names below are in the `PearsonClassification` namespace.

| TeX claim / label | Checked counterpart |
| --- | --- |
| Feature measurability in `ass:setup` | `Moments.feature_stronglyMeasurable` bridges measurable separable-Hilbert features to strong measurability |
| Independent sampling in `ass:setup` | `BoundedDifferences.law`, `observations_map`, `observation_map`, `observations_independent`, `independent_samples_law`; `Statistical.batchLaw` has the reference/incoming marginals |
| Population/empirical moments, `eq:pop-moments`, `eq:emp-means`, `eq:emp-moments` | `Moments.mean`, `secondMoment`, `empiricalMean`, `empiricalSecondMoment`; bounded-feature integrability is proved |
| Reproducing and kernel identities | `Moments.rkhs_reproducing`, `rkhsFeature_inner`, `rkhsFeature_norm_sq`, `kernel_uniform_bound`, `evaluation_uniform_bound` |
| `eq:quadratic`, `0 ≤ T_P ≤ K I` | `Moments.secondMoment_quadratic`, `secondMoment_positive`, `secondMoment_upper` |
| `0 ≤ T_hat ≤ K I` | `Moments.empiricalSecondMoment_positive`, `empiricalSecondMoment_quadratic`, `empiricalSecondMoment_upper` |
| Inverses and scores, `eq:A`, `eq:D`, `eq:Dhat` | `Operators.regularizedEquiv`, `coercive_isUnit`, `regularized_coercive`; `Moments.populationScore`, `empiricalScore`; `Operators.inverse_operator_norm_bound` |
| Decision rule, `eq:decision` | Error events use `score ≤ η` for acceptance; `acceptedReferences` implements the bank acceptance set |
| Confidence radii, `eq:radii` | `meanRadius`, `operatorRadius` are the exact formulas |
| Mean concentration, `lem:mean`, `eq:mean-event` | `Statistical.mean_concentration`, `mean_high_probability` |
| Variance/diameter, `eq:feature-variance`, `eq:feature-diameter` | `Moments.feature_variance_identity`, `feature_variance_upper`, `feature_diameter` |
| Empirical variance, `eq:delta-second-moment` | `Sampling.sumStat_variance`, `centeredSum_variance_exact`, `centeredSum_variance`; `Statistical.mean_error_second_moment` gives the exact two-window formula |
| Mean exponential tail, `eq:mean-tail` | `Sampling.weighted_mean_tail`, `Statistical.mean_tail` |
| Operator concentration, `lem:operator`, `eq:op-event` | `Statistical.operator_concentration`, `operator_high_probability` |
| Tensor identities, `eq:Vnorm`, `eq:Vdiameter` | `HilbertTensor.norm_liftRankOne`, `inner_liftRankOne`, `rankOne_diameter` |
| Operator empirical variance, `eq:operator-variance` | `Statistical.operator_tensor_variance`, `Sampling.centeredSum_variance`; `tensorMean_toOperator` and `tensorEmpirical_toOperator` identify the actual moment operators |
| Operator exponential tail, `eq:op-tail` | `Statistical.operator_tensor_tail`; `operator_error_le_tensor` transfers tensor error bounds to the operator norm |
| Null deterministic bound, `eq:null-domination` | `null_quadratic_upper_bound`, `Operators.quadraticScore_nonneg` |
| Null probability/error, `thm:null`, `eq:null`, `eq:U`, `eq:false-separation` | `Statistical.null_high_probability`, `null_error`; `nullBound` |
| Joint confidence event, `eq:joint-event`, `eq:joint-probability` | Union-bound step in `alternative_high_probability_of_concentration`, discharged by `Statistical.alternative_high_probability` |
| Inverse comparison, `lem:inverse`, `eq:inverse-order` | `Operators.perturbation_order`, `inverse_comparison` on actual operators, with no commutativity premise |
| Non-merging bound, `thm:alternative`, `eq:alternative`, `eq:L`, `eq:false-merging` | `nonmerging_quadratic_lower_bound`, `alternativeBound`; `Statistical.alternative_high_probability`, `alternative_error`, including the `α/2` split |
| Static threshold, `cor:threshold`, `eq:threshold-interval`, `eq:errors` | `Statistical.static_threshold`; endpoint conventions are `U ≤ η < L` |
| Nonempty interval, `prop:nonempty`, `eq:separation-condition` | `threshold_gap_iff`, `exists_static_threshold_iff`, `calibrated_threshold_iff` |
| Explicit calibration, `eq:eta`, `eq:explicit-delta` | `nullBound_explicit`, `separationRequirement_eq`, `explicit_calibration_iff`, `confidenceMultiplier_half` |
| Fixed-parameter convergence | `meanRadius_tendsto_zero`, `operatorRadius_tendsto_zero`, `nullBound_tendsto_zero`, `alternativeBound_tendsto_separation`, `eventually_nonempty_threshold` |
| Rates with varying regularization | `meanRadius_scaled`, `operatorRadius_scaled` are exact identities; `meanRadius_isBigO`, `operatorRadius_isBigO` give the stated rates |
| Batch size, `cor:sample-size`, `eq:batch-size` | `sufficient_batch_size` |
| Fixed bank, `cor:bank`, `eq:known-classification`, `eq:unknown-classification` | `Statistical.bank_known`, `bank_unknown`; generic union bounds are in `BoundsExtendedProbability` |

## Appendices

| Appendix content | Checked counterpart / qualification |
| --- | --- |
| Bochner foundations and passage through expectations | `Moments.feature_integrable`, `moment_integrable`, `normSquare_integrable`, `mean_inner`, `secondMoment_inner` |
| Score/MMD bounds and zero scores | `Moments.score_MMD_bounds`, `populationScore_zero_iff`; positivity follows in `Statistical.populationScore_pos` when means differ |
| General Hilbert variance identity | `Moments.variance_identity` takes integrability of the feature and its squared norm, without a pointwise bound |
| Independent empirical averages | `Moments.independent_centered_inner`, `Sampling.sumStat_variance`, `iid_mean_variance`; the assembled finite-sum results use bounded features, as required in the main text |
| Two-point indicator-kernel example | `Examples.orthogonalFeature_inner`, `orthogonal_mean`, `orthogonal_point_error`, `orthogonal_variance`, `orthogonal_variance_upper`, `orthogonal_example_numbers`; `orthonormal_feature_exists` supplies actual vectors |
| Rank-one Hilbert-space identities and operator domination | `HilbertTensor.norm_completed_rankOne`, `inner_completed_rankOne`, `toOperator_norm_le`, `error_domination`; see the tensor-model qualification below |
| Generic McDiarmid and coordinate sensitivities | `BoundedDifferences.mcdiarmid`, `global_diameter`, `subgaussian_of_hasDifferences`; `Sampling.sumStat_change`, `sumNorm_differences`, `centeredFamily_diameter` check replacements and all input coordinates |
| Weighted norm and noncommuting operator order | `Operators.inverse_comparison`, `weightedNorm_triangle`, `weightedNorm_upper`; completing squares replaces the manuscript's spectral proof |
| Coupled-estimator example | `Examples.scalarScore_eq_empiricalScore`, `coupled_score_values`, `coupled_score_expectation`, `coupled_failed_factorization`, `coupled_off_diagonal_mass` check the actual score, integrals, and probabilities exactly |
| Positive finite-family margin | `Statistical.finite_family_separation`, from injective mean embeddings at fixed regularization |
| Mixture calculation and score tending to zero | `Moments.mean_mixture`, `mixture_score_scaling`, `mixture_score_tendsto_zero`, for any feature map satisfying the assumptions |
| Population and empirical variational problems | `Moments.populationScore_variational`, `population_unique_maximizer`, `empirical_variational_objective`, `empiricalScore_variational`, `empiricalScore_unique_maximizer`; `Operators.completing_square` proves the quadratic identity |
| Threshold expansion | `separationRequirement_eq`, `confidenceMultiplier_half`, `explicit_calibration_iff` |
| Actual score-event measurability | `Statistical.continuous_featureScore`, `regularized_inverse_eq`, `batchScore_eq_vectorScore`, `batchScore_measurable` |

## Remaining scope

1. **Gaussian kernel background.** `Examples.gaussianKernel_bounds` and
   `gaussianKernel_diagonal` prove pointwise positivity and the diagonal
   value `1`. Positive definiteness and characteristicness of the Gaussian
   kernel are not proved here. Consequently, a Gaussian-specific application
   still needs a kernel/RKHS construction and, if used to obtain separation,
   a mean-embedding injectivity theorem on the chosen distribution domain.
   None of the generic classification proofs assumes these facts as axioms.
   The mixture result already applies to any such kernel once its feature
   assumptions are established.
2. **Literal Hilbert–Schmidt identification.** The operator-concentration
   proof constructs the completed Hilbert tensor square and a contractive
   map into bounded operators. It proves every identity, expectation bridge,
   diameter bound, and domination used by the main argument in that model.
   There is no separate theorem identifying it isometrically with the
   appendix's class of all operators defined by a basis-square-sum condition,
   or reproducing that particular Parseval proof. The main concentration
   theorem does not depend on assuming this identification.
3. **Broader appendix generality and proof routes.** The general variance
   identity is proved for square-integrable features, while the assembled
   independent finite-sum variance theorem uses bounded summands. The text's
   general unbounded square-integrable i.i.d.-average statement is not
   separately assembled. Spectral inverse-square-root constructions and
   qualitative explanatory prose are not individually formal declarations;
   the required operator-order and weighted-norm conclusions are proved
   directly.

There are no open concentration or operator-invertibility premises in the
final main theorems. Nevertheless, these remaining items prevent claiming
that every mathematical assertion in the entire TeX file has been certified.

## Organization

* `BoundsExtendedOperators`, `Moments`, `Tensor`, `Foundations`, and
  `Appendices`: deterministic functional analysis and score construction.
* `BoundsExtendedMcDiarmid`, `Sampling`, `SamplingModel`, `Variance`, and
  `Tails`: independent sampling and derived concentration.
* `BoundsExtendedCore`, `Calibration`, and `Rates`: scalar bounds,
  calibration, and asymptotics.
* `BoundsExtendedProbability`: reusable conditional/union-bound lemmas.
* `BoundsExtendedStatistical`, `Guarantees`, and `Measurability`: actual
  statistical model and final guarantees.
* `BoundsExtendedExamples`: finite probability examples and Gaussian envelope.
* `BoundsExtendedAudit`: proof-dependency audit.

The manuscript was clarified to require `M ≥ 1` for `α/M` calibration and
positive tails/nonzero squared sensitivity sum in its displayed McDiarmid
formula. These are explicit mathematical domain conditions, not inferred
facts about a particular dataset.
