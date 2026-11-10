# Pearson and MMD open-world change-point detection

Two online detectors compare adjacent reference and analysis windows, confirm
persistent changes, and recognize returning regimes from fresh post-alarm
representatives. A regime is a stretch of observations drawn from the same
distribution. Each experiment compares both detectors on the same stream.

## Setup

Use Python 3 with a virtual environment. Run these commands from the repository
root (the directory containing this README):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy scipy matplotlib pandas scikit-learn
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
The repository currently has no dependency lockfile; the command above installs
the dependencies without pinning versions.

Start with the Gaussian example, which generates its data locally:

```bash
python3 main_mmd_pearson.py
```

It prints detection and recognition metrics and opens Matplotlib figures.
Experiment mains print results and display plots without saving files by default.

Run the individual main you are editing while tuning parameters.
MNIST, PAMAP2, and WISDM require internet access for their initial dataset downloads.

## Run the experiments

```bash
python3 main_mmd_pearson.py  # ICASSP §5.1: Gaussian reference example
python3 main_T1.py           # §5.2: distinct means, common variance
python3 main_T2.py           # §5.2: common mean, distinct variances
python3 main_T3.py           # §5.2: common mean and variance
python3 main_mnist.py        # §5.3: sequences A and B
python3 main_pamap2.py       # Real PAMAP2 sensor recordings
python3 main_wisdm.py        # Real WISDM Activity Recognition v1.1 recordings
```

Run one command at a time for the experiment you want. T1, T2, and T3 are
independent entry points: none imports or executes another tier's main. There is
no master T script to run first; the old `main_T_exp.py` shortcut has been removed.
The shared modules provide reusable functions, without starting experiments when
imported.

Before running, edit `build_models()` in the selected main file. It contains
explicit `OWR_CPD(...)` and `PearsonOWRCPD(...)` constructor calls with every model
parameter visible: window sizes, confirmation count, thresholds, `gamma`,
`factor`, `dynamic_gamma`, and Pearson's `reg_lambda` and `normalize_distance`.
The named constants at the top supply the default experiment settings. You can
change those constants or edit either constructor argument independently.
Values marked as arbitrary placeholders are starting values, not tuned settings.

For example, in `main_mmd_pearson.py`, set `gamma=0.25` and
`dynamic_gamma=False` in a model's constructor to use a fixed bandwidth. With
`dynamic_gamma=True`, the model estimates gamma from each representative batch;
`factor` controls the scale of that estimate.

The shared runner receives your model instances without constructing or
reconfiguring them. Each repeated stream and MNIST sequence gets a fresh pair.
`MAX_DELAY` sets the common scoring tolerance independently of model parameters.
Returned results record each model's actual constructor settings under
`config.model_parameters`, including its gamma before streaming updates.

For the Gaussian, T1/T2/T3, and MNIST experiments, disable plotting with
`main(show=False)`. For example:

```bash
python3 -c "import main_T1; results = main_T1.main(show=False)"
```

The returned dictionary contains the models and metrics (inside `runs` for T1/T2/T3
and `sequences` for MNIST). PAMAP2 and WISDM expose `run_experiment()` instead;
these functions accept `show=False` and return the same models/metrics dictionary.
For a terminal-only run of either activity script, use a non-interactive backend:

```bash
MPLBACKEND=Agg python3 main_wisdm.py
```

Every entry point uses the shared runner and evaluator. The paper benchmark
settings are listed below; activity configurations are in their respective scripts.

| Experiment | Segment size | n | m | MMD detection threshold | MMD novelty threshold |
|---|---:|---:|---:|---|---|
| Gaussian reference | 300 | 150 | 40 | 0.12 | 0.90 |
| T1 | 700 | 150 | 40 | 0.1 (placeholder) | 0.63 |
| T2 | 700 | 300 | 100 | 0.1 (placeholder) | 0.99 |
| T3 | 700 | 600 | 200 | 0.1 (placeholder) | 0.99 (placeholder) |
| MNIST A and B | 300 | 150 | 40 | 0.06 | 0.96 |

All current experiment configurations use six consecutive exceedances for confirmation. T1/T2/T3 each
run three independent streams (seeds 42, 43, 44), with eight true returns per
stream. Each seed prints its own five-metric MMD/Pearson table and produces its
own score and regime-ID figures, with the seed in each title. All three runs
finish before the plot windows block; pooled results print afterward.

Running `python3 main_T1.py` (or T2/T3) prints the individual and pooled
tables and shows the figures without writing reports or plot files. To run
without opening windows:

```bash
python3 -c "import main_T1; main_T1.main(show=False)"
```

The T1/T2/T3 data generators use the `MEANS` and `STDS` configured in each
main. These population parameters stay fixed across seeds. The tier constraints
are validated: T1 has four distinct means and one common std; T2 has one common
mean and four distinct stds; T3 has one common mean and one common std.
Finite batches can have different sample means/stds.

With `RANDOMIZE_FAMILIES = True` (the default), each seed assigns Gaussian,
Laplace, exponential, and uniform to regime IDs 1–4 through a random permutation.
Every family occurs exactly once among the four regimes. The same seed gives
the same assignment across T1/T2/T3. That assignment stays fixed on all returns;
each episode draws fresh observations. The chronological episode-label sequence
is unchanged. Both models see the same generated stream, and the family mapping
is printed and retained as `config.stream_metadata` in each run's returned results.
Setting `RANDOMIZE_FAMILIES = False` restores the historical fixed assignment
(1=Gaussian, 2=Laplace, 3=exponential, 4=uniform) and sampling sequence.

For requested population mean `mu` and std `sigma`, sampling uses:

| Family | Parameters / transformation |
|---|---|
| Gaussian | mean `mu`, std `sigma` |
| Laplace | location `mu`, scale `sigma / sqrt(2)` |
| Exponential | `mu - sigma + Exponential(scale=sigma)` |
| Uniform | lower `mu - sqrt(3)*sigma`, upper `mu + sqrt(3)*sigma` |

The exponential is shifted so it can share any requested mean with the other
families while retaining std `sigma`. No batch is normalized using future data
to force exact empirical moments.

## Choosing parameters

Settings above, except those marked as placeholders, follow `icassp_2027.pdf`,
sections 5.1–5.3.
The PDF omits tier-specific detection thresholds, precise T1/T2 population
parameters, seeds, and the full precision of T3's printed novelty threshold
`1.00`. The Pearson manuscript supplies no experimental parameter table.
Consequently these scripts cannot promise exact reproduction of the paper's
counts. Implementation choices are explicitly marked in each main.

All detection and novelty thresholds are chosen manually in the main file for
the current experiment:

- `MMD_ETA`: detection threshold for squared MMD; higher scores trigger changes.
- `MMD_NU`: minimum coherence for recognizing a stored regime; higher is closer.
- `PEARSON_ETA`: detection threshold for the Pearson-type score.
- `PEARSON_NU`: maximum Pearson-type score for recognizing a stored regime;
  lower is closer.
- `REG_LAMBDA`: Pearson operator regularization, with Gram ridge `n * lambda`.

Previously unspecified thresholds have arbitrary numeric placeholders: MMD
detection `0.1`, T3 MMD novelty `0.99`, Pearson detection `0.4`, and Pearson
novelty `0.7`. These are not calibrated or theoretically selected. Replace them
with your chosen values in each main. There is no automatic threshold estimation
or fallback. Setting a threshold to `None` produces a clear error.
The theoretical notes support your choice of parameters; the scripts do not use
them to choose values. Window sizes and regularization affect score scales, so
consider them together when choosing thresholds.

Reusable synthetic generators, including the four T1/T2/T3 distribution families,
live in `data/generation.py`. The Gaussian reference main also contains its own
experiment-specific sampling code. Each tier main defines its parameters and loops over its
seeds. `experiments/runner.py` shares execution, plotting, and reporting only;
`experiments/evaluation.py` computes metrics.

MNIST reserves disjoint per-digit pools for fitting PCA (500 images) and
evaluation. PCA is fitted once on the preprocessing pool; both models use the
same 50-dimensional coordinates and Euclidean RBF kernel, without extra per-image
L2 normalization. No evaluation image is reused within a sequence or across A and
B. `fetch_openml` downloads the existing MNIST dataset from OpenML and caches it in
`data/openml/`; it does not generate synthetic observations. No images are
reserved for threshold calibration.

## Repository folders and files

The root scripts choose the experiment, data, and parameters. `data/` prepares
observations, `model/` implements the detectors, and `experiments/` shares running
and scoring utilities. `proofs/` contains the
mathematical background.

### Root: experiment entry points and project information

| File | Objective and behavior |
|---|---|
| [main_mmd_pearson.py](main_mmd_pearson.py) | Gaussian reference example with four regimes in the sequence `1 → 2 → 3 → 2 → 1 → 4`. Generates observations, constructs both models explicitly in `build_models()`, and invokes the shared comparison runner. Importing it does not run the experiment. |
| [main_T1.py](main_T1.py) | Compares Gaussian, Laplace, exponential, and uniform families with distinct means and a common variance. Owns its parameters and runs its three seeds independently. |
| [main_T2.py](main_T2.py) | Compares the same families with a common mean and distinct variances. Owns its parameters and runs its three seeds independently. |
| [main_T3.py](main_T3.py) | Compares the same families with common population mean and variance, so differences concern distribution shape. Owns its parameters and runs its three seeds independently. |
| [main_mnist.py](main_mnist.py) | Downloads MNIST through `fetch_openml`, separates PCA and evaluation images, fits a shared PCA representation, builds digit sequences A and B, and compares both detectors. Repeated digits consume fresh images, including across sequences. |
| [main_pamap2.py](main_pamap2.py) | Builds a recurring activity stream from PAMAP2, currently using 40 sensor features and 1,500 observations per segment. Fits standardization on the initial reference window, optionally applies PCA, and runs and plots both detectors. |
| [main_wisdm.py](main_wisdm.py) | Loads real WISDM Activity Recognition v1.1 accelerometer recordings from one participant. Builds recurring episodes from distinct contiguous blocks, optionally downsamples with causal filtering, fits preprocessing on the initial reference window, and runs both detectors on the same measurements. |
| [README.md](README.md) | Setup, run instructions, repository map, parameter meanings, and evaluation conventions. |
| [.gitignore](.gitignore) | Excludes Python caches, local environments, generated artifacts, and downloaded MNIST/PAMAP2/WISDM data from version control. |

### `data/`: generate observations and prepare downloaded data

| File or generated directory | Objective and behavior |
|---|---|
| [generation.py](data/generation.py) | Reusable synthetic generators. `generate_mean_shift_data()` creates Gaussian mean shifts; `generate_regime_stream()` varies Gaussian means and standard deviations; `sample_family()` samples the four T distribution families; `generate_moment_matched_stream()` validates tier moments, assigns families to regime IDs per seed, and assembles the recurring sequence and true change points. Optional metadata records the realized assignment and population moments. The first two helpers remain available but are not currently called by the mains. |
| [manage_pamap2.py](data/manage_pamap2.py) | Downloads and extracts PAMAP2 if needed, reads `Protocol/subject101.dat`, maps activity labels, and takes distinct blocks from contiguous activity runs. Uses forward filling within each run and discards leading missing values. Provides a three-channel hand-acceleration loader and a 40-feature loader excluding invalid orientation channels. Raises an error if the requested distinct blocks are unavailable. |
| [manage_wisdm.py](data/manage_wisdm.py) | Downloads the official WISDM AR v1.1 archive and preserves its documentation. Parses measured acceleration rows, selects one participant, and extracts disjoint blocks without crossing activity/participant changes, invalid records, backward timestamps, or configured timestamp gaps. Records source positions, timestamps, and a raw-file SHA256. |
| `openml/` | Generated MNIST download cache managed by scikit-learn. OpenML supplies an existing dataset; this directory is not a generator or Python module. |
| `PAMAP2_Dataset/` | Generated directory holding the extracted PAMAP2 recordings. The loader skips downloading when this directory already exists. |
| `wisdm/` | Cached official WISDM archive, raw readings, and accompanying documentation. No generated observations. |

The download directories are created as needed and ignored by Git. Dataset
files themselves are not included in the repository. For PAMAP2, reduce the block
size or requested activity repetitions if the selected recording cannot supply
enough distinct contiguous blocks.

PAMAP2 and WISDM use the T experiments' recurrence order,
mapped from T regime IDs 1–4 to their existing activity IDs 0–3:
`0 → 1 → 2 → 3 → 1 → 0 → 3 → 2 → 0 → 3 → 1 → 2`.
Each activity appears three times, giving 12 segments and 11 true changes.
Returns consume distinct observations; segment lengths and preprocessing remain
configured in each activity script.

Edit `SEQUENCE` directly in `main_pamap2.py` or `main_wisdm.py`; the two mains
keep independent literal sequences. `SEGMENT_LENGTH` in each main controls raw
samples per activity occurrence (WISDM defaults to 700 measured rows with no
downsampling). `N` controls the reference and
post-alarm recognition window; `M` controls the analysis window. These window
sizes apply to every activity and are measured after preprocessing/downsampling.
PAMAP2's `ACTIVITY_MAPPING` maps dataset activity IDs to the truth IDs in its
sequence. WISDM's `ACTIVITY_MAPPING` maps original activity names to truth IDs.
Detector-assigned IDs are learned online and can differ from truth IDs.

The WISDM benchmark uses the [official Fordham Activity Recognition v1.1 release](https://www.cis.fordham.edu/wisdm/dataset.php),
not the separate smartphone/smartwatch release. Each observation contains the
phone accelerometer's x/y/z readings. In `main_wisdm.py`, edit `SUBJECT_ID`,
`SEQUENCE`, `SEGMENT_LENGTH`, `ACTIVITY_MAPPING`, and `DOWNSAMPLE_RATE` directly.
The defaults use participant 1, activity IDs 0=Walking, 1=Jogging, 2=Upstairs,
3=Downstairs, and three non-overlapping 700-row episodes per activity. The mapping
also offers 4=Sitting and 5=Standing, but not every participant has every activity
or enough measurements for every requested sequence. Insufficient data raises
an error; the loader never changes participants, pads, replays, interpolates, or
generates observations to satisfy a request.

WISDM's documented sampling rate is nominally 20 Hz; original timestamps can be
irregular. Source order and equal timestamps are preserved. Runs break at invalid
rows (including zero timestamps), backward timestamps, activity/participant
changes, or gaps exceeding `MAX_GAP_SECONDS` (default 0.5 seconds). Optional
downsampling applies a causal FIR filter before retaining rows; the default
factor 1 uses the measured rows directly. Scaling and optional PCA fit only on
the initial reference window. The returned `config.dataset` records the raw-file
hash, activity mapping, participant, and source record/timestamp ranges for every
episode. The Gaussian placeholder generator has been removed; prior results from
it are synthetic results and must not be reported as WISDM performance.

PAMAP2's `DOWNSAMPLE_RATE` controls its effective sampling rate from the original
100 Hz: use 1 for 100 Hz, 2 for 50 Hz, 4 for 25 Hz, or 5 for 20 Hz. For factors
above one, a causal FIR low-pass filter suppresses aliasing before decimation.
Filtering uses only sensor observations, with no future samples or activity-label
resets; filter latency remains in the measured detection delay. Labels are
subsampled and change indices recomputed at the model's sampling rate.
`SEGMENT_LENGTH` still counts raw 100 Hz samples, preserving activity durations.
`N`, `M`, `K_CONF`, and `MAX_DELAY` count observations at the reduced rate, so
their durations in seconds increase when their values are kept fixed. The script
prints the effective frequency and model window durations, and rejects settings
where initialization would extend beyond the first activity episode. Lower rates
can suppress high-frequency noise, but their effect on detection must be measured.

### `model/`: online detection and regime memory

| File | Objective and behavior |
|---|---|
| [owcl_mmd.py](model/owcl_mmd.py) | Defines `OWR_CPD`. Detects changes with biased empirical squared MMD using a Gaussian RBF kernel. Recognizes regimes using normalized similarity between kernel mean embeddings. Stores regime representatives and their bandwidths. Includes distance, bandwidth, and parameter-validation helpers. |
| [owcl_pearson.py](model/owcl_pearson.py) | Defines `PearsonOWRCPD`. Uses a directed, regularized RKHS Pearson-type discrepancy for detection and recognition. Maintains the inverse regularized Gram matrix recursively, periodically rebuilding it to limit numerical drift. Supports optional observation normalization and stores each representative's operator statistics and bandwidth. |

Both detectors process one observation with `update(x)`. They first collect a
reference window, fill an analysis window, monitor adjacent sliding windows, and
confirm a change after consecutive threshold exceedances. After an alarm they
collect a new representative, assign an existing or new regime ID, and restart
monitoring. Their attributes expose score histories, alarm times, regime IDs, and
recognition events. The Pearson-type score is not assumed to equal unrestricted
classical Pearson divergence.

### `experiments/`: shared execution and evaluation

| File | Objective and behavior |
|---|---|
| [runner.py](experiments/runner.py) | Accepts models constructed by the experiment main, validates manual thresholds, expands segment labels, feeds identical observations to both models, records their actual settings, prints summaries, aggregates repeated runs, and plots scores and regime IDs. It does not construct models or override their parameters. |
| [evaluation.py](experiments/evaluation.py) | Scores confirmed alarms and timely matched-alarm assignments using a chronological one-to-one mapping, excludes warm-up from assignment scores, and pools counts and delays across runs. |

These are utilities rather than separate experiments. There is no calibration
module: parameter selection belongs to the main you are running. No
`__init__.py` is required for these directory imports when running from the
repository root with Python 3.

### `proofs/`: mathematical support for parameter choices

| File | Objective |
|---|---|
| [proofs.pdf](proofs/proofs.pdf) | Concise pairwise classification analysis: concentration, same-regime upper bounds, separated-regime lower bounds, and an admissible Pearson-type threshold interval. |
| [bounds_extended.pdf](proofs/bounds_extended.pdf) | Extended operator-based derivations, including sample size, regularization, and fixed memory-bank classification guarantees. |
| [pearson_geometric_bounds_extended.pdf](proofs/pearson_geometric_bounds_extended.pdf) | Geometric assignment bounds based on embedding errors, Markov-based probability guarantees, comparisons with operator-aware bounds, and limitations of the certificates. |

These PDFs are reference documents. No experiment reads them or derives its
parameters automatically from them.

## Bandwidth and online procedure

Each initial or replacement representative is built in this order:

1. Collect its n observations.
2. Compute `gamma = 1 / (2 * median(pairwise squared distances))` using distinct
   pairs, with a small floor for degenerate distances.
3. Compute its kernel mean embedding and, for Pearson, its uncentered empirical
   second-moment operator through the regularized Gram inverse.
4. Store the representative with that bandwidth.

Every recognition candidate uses its own stored bandwidth for the entire
comparison. A fresh batch's bandwidth is computed before its embedding/operator
and stored when it becomes the assigned representative, including on recall.
The active bandwidth is then held fixed through sliding-window monitoring until
another representative is collected. Adaptive bandwidths are the chosen
experimental policy; the Pearson manuscript's fixed-kernel concentration bounds
do not directly establish guarantees for this adaptive selection.

The first score is evaluated at sample count `n + m`, and monitoring resumes at
`t_alarm + n + m`. The n observations immediately after an alarm form the
recognition batch. Post-alarm collection does not itself guarantee a homogeneous
batch, particularly with T3's long reference window and 700-sample segments.

## Evaluation

All entry points use `experiments/evaluation.py`. Models receive only observations
via `update(x)`; true boundaries and labels are used only for scoring and plots.
Each benchmark stream is generated/preprocessed once and supplied unchanged to
both models. Repeated T1/T2/T3 streams use paired seeds 42, 43, 44; model random
state is reset to the same seed. Other configured benchmarks use a single stream.

True change times are zero-based indices of the **first sample in the new
episode**. Model logs store one-based sample counts. The evaluator converts both
confirmed alarm counts and assignment counts by subtracting one. Detection delay
is `alarm_index - change_index`: confirmation at the first changed sample has
zero delay. Assignment latency never changes the alarm timestamp.

The inclusive maximum alarm delays, fixed before either model is scored, are:

| Benchmark | Maximum delay (samples seen by the model) |
|---|---:|
| Gaussian reference | 46 |
| T1 | 46 |
| T2 | 106 |
| T3 | 206 |
| MNIST A and B | 46 |
| PAMAP2 | 106 |
| WISDM Activity Recognition v1.1 | 56 (original observations by default) |

Each is `MAX_DELAY = M + K_CONF`, an evaluation choice rather than an ICASSP
setting. Matching takes the earliest unused confirmed alarm at or after a change,
within the tolerance and strictly before the next change or stream end. At most
one alarm matches each change. Every unmatched monitoring alarm, including extra
alarms after a match, is false. Initial reference collection is warm-up; alarms
before its completion are ignored. Later monitoring alarms in the initial episode
can be false alarms. Detection is reported as matched changes / all changes after
the initial episode, and mean delay is computed over matched alarms only.

Assignment uses only the decision from the episode's matched alarm. It must be
available before the episode ends, using the actual decision sample. A carried
ID, missed alarm, missing decision, or late decision earns no credit. There is no
additional batch-purity requirement. The evaluator initializes a separate mapping
for each model/run with the warm-up regime's ID and excludes the initial episode
from both assignment denominators. That regime's later appearances are returns.

The truth-to-model-ID mapping is chronological and one-to-one. A regime without
a valid mapping succeeds by creating a fresh, unclaimed ID at a timely matched
assignment. Once mapped, it must reuse that ID; merging with another mapped
regime or creating a duplicate ID fails. Established mappings never change.
First-encounter discovery includes **all** true first appearances after warm-up
in its denominator. Return assignment includes **all** ground-truth return
episodes, even if the regime was not successfully discovered earlier. Creating
its first valid ID on a return earns credit for that return only. Earlier failures
remain failures; mappings never use future decisions to revise scores.

The sanity check `truth=1–2–1–3–2–1`, `IDs=1–1–1–2–3–1` scores discovery **1/2**
and return assignment **3/3**, with mapping `{1: 1, 3: 2, 2: 3}`.

Reports show all five metrics side by side. Repeated runs sum detection and
assignment numerators/denominators and pool every matched detection delay;
per-run means/rates are never averaged. Empty rate denominators and delay with
no matches display `N/A` (`null` for delay in JSON).

Thresholds and regularization are preserved from the configured benchmarks,
including arbitrary placeholders; this is not a tuned performance claim. In
particular PAMAP2 retains Pearson `lambda=20`, `eta=0.80`, and `nu=0.71`.
The detection score is bounded by `1/lambda=0.05`, so it cannot alarm at that
threshold. Its runner explicitly disables the usual score-bound validation to
measure the original configuration without silently replacing parameters.
WISDM now uses real recorded measurements. Its manually configured thresholds
have not been calibrated for this dataset; replacing the data source does not
establish a performance advantage.
