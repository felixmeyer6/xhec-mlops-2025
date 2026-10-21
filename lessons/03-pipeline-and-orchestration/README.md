# Machine Learning Pipelines and Orchestration with Prefect

Be sure to have your virtual environment synced with all the project dependencies by running `uv sync` in the root of the repository.

Please follow the instructions in the [notebook](orchestration.ipynb).

## Completed implementation

The notebook imports the implemented scripts in `lib/` so interactive runs and
deployments use the same code. Download the three taxi parquet files with the
introduction notebook first. All commands below run from this lesson directory,
using the repository's Python 3.11 environment and pinned Prefect **2.13.7**.

### Run the workflows

```bash
uv run python -m lib.workflows train
uv run python -m lib.workflows predict
```

Training fits a DictVectorizer and linear regressor on January, then evaluates both
training and February data using identical inclusive 1–60 minute duration filters.
The fitted vectorizer is never refitted on February or inference data. It is saved
with its matching model, feature schema, sklearn version, data paths and RMSEs in
`models/orchestration/taxi_model.pkl` at the repository root. This single bundle is
atomically replaced, so prediction can never see half of a weekly update.

Prediction loads the saved pair independently, reads March booking features only
(no timestamps or target required), preserves row order and writes
`models/orchestration/predictions.npy`. Read it with
`numpy.load(path, allow_pickle=False)`. All data and generated artifacts remain
git-ignored. Load model pickles only from trusted training runs.

The flows also accept explicit file paths:

```bash
uv run python -m lib.workflows train --train /path/train.parquet --test /path/test.parquet --artifacts /path/model.pkl
uv run python -m lib.workflows predict --input /path/input.parquet --artifacts /path/model.pkl --output /path/predictions.npy
```

The Python API additionally supports in-memory model/vectorizer handoff when no
artifact path is supplied. Named tasks and a preprocessing subflow expose explicit
dependencies in Prefect. File I/O retries twice with a one-second delay; deterministic
preprocessing/model errors do not retry. Explicit artifacts, rather than Prefect's
result persistence, provide the durable handoff between separately scheduled runs.

**Model limitation:** linear regression can predict negative or unusually large
durations. Outputs are not silently clipped; batch logs count estimates outside
[1, 60] minutes. Inspect these before treating them as useful booking estimates.

### UI and scheduled deployments

Without `PREFECT_API_URL`, one-off workflows use Prefect's ephemeral local API.
For the UI and continuously served schedules, start the server in one terminal:

```bash
uv run prefect server start --host 127.0.0.1
```

In another terminal, from this lesson directory:

```bash
export PREFECT_API_URL=http://127.0.0.1:4200/api
uv run python -m lib.workflows train
uv run python -m lib.deployment --check
uv run python -m lib.deployment
```

The initial training run ensures the first hourly prediction has a model bundle.
`--check` only constructs and prints schedules; without it, `serve()` registers
both deployments and stays running to execute scheduled work:

| Deployment | UTC schedule | Cron |
| --- | --- | --- |
| `weekly-training` | Monday at 00:00 | `0 0 * * 1` |
| `hourly-prediction` | Every hour at minute 0 | `0 * * * *` |

Visit http://127.0.0.1:4200/dashboard to inspect task logs, dependencies and states.
Export the same API URL before launching the notebook kernel to make its runs visible
there too. Keep both server and runner processes alive. Stop the runner with Ctrl+C;
it pauses schedules using an independent cleanup event loop while the server remains
reachable. If the runner is forcibly killed or loses API connectivity, manually
pause its schedules in the UI. Do not call blocking `serve()` in Run All.

These local example schedules reuse the fixed historical input files; they do not
fetch new hourly data. Each prediction run replaces the same output array. A real
pipeline needs fresh input discovery, versioned outputs, model-validation/promotion
policy and retention. Atomic artifact replacement prevents partial reads, but does
not guarantee the hourly run waits for a simultaneous weekly retraining: it uses
whichever complete model bundle is available when it starts. The fixed data/artifact
paths must exist on the runner machine; this is not a cloud work-pool deployment.

### Tests

From this lesson directory, using a disposable Prefect home (to isolate test runs
from your own profiles/database):

```bash
PREFECT_HOME="$(mktemp -d)" PREFECT_API_URL='' PREFECT_SERVER_ANALYTICS_ENABLED=false \
  uv run python -m unittest discover -s tests -v
```

Tests cover filtering boundaries, input non-mutation, missing/unknown categories,
inference without targets, no inference refitting, empty data, atomic artifact
updates, prediction file parity, real Completed/Failed flow states and UTC schedules.
One test intentionally creates a Failed flow to verify failure reporting.
