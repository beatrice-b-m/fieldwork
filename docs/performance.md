# Large dataframes and runtime feedback

Fieldwork keeps exact source identity and population accounting while reducing
repeated scans, scalar serialization, row-by-row grouping, and temporary arrays.
Default analytical results and search budgets are unchanged. No sampling or
additional required runtime dependency is introduced.

## See what is running

```python
import fieldwork as fw

overview = fw.explore(df, progress=True)
```

`progress=True` shows one updating terminal bar or IPython notebook display.
Redirected output receives newline-delimited updates on stderr. Each phase shows
its name, elapsed time, and work completed where a total is known. Phases include
fingerprinting, encoding, path search, dependency tests, grain views, and pattern
summaries. Completion of one phase does not imply completion of the overview.
The default remains silent.

All public dataframe analyses accept keyword-only `progress`, `cancel`, and
`timeout` (typed once as `fieldwork.typing.Runtime`), including `profile`, foundation tools (`levels`,
`census`, `grain`, `pairs`, `joint_counts`, `infer_schema`), `Recipe.run`,
`Path.census`, source inspection/selection/recomputation, and scope creation and
refinement. Rendering uses saved results and does not run dataframe analysis.

For a custom UI, pass a callable instead of `True`:

```python
def report(event):
    print(event.phase, event.completed, event.total, event.status)

analysis = fw.discover_dependencies(df, progress=report)
```

Callbacks receive immutable `ProgressEvent` objects with `operation`, `phase`,
`phase_id`, `parent_id`, `completed`, nullable `total`, `unit`, `elapsed_seconds`,
`phase_elapsed_seconds`, nullable `estimated_remaining_seconds`, nullable `detail`,
and `status`. IDs identify nested phase instances within one call. Status is
`started`, `running`, `completed`, `cancelled`, or `failed`. The operation names
are descriptive; use IDs and parent IDs to track nested work.

Running updates are throttled to at most five per second across the operation;
phase boundaries are always reported. Callbacks run synchronously. Keep them
quick and do not mutate the dataframe during analysis. An exception raised by a
callback stops the analysis and propagates unchanged; a broken callback is not
called again during cleanup.

The displayed ETA estimates **only the current phase**, after at least four
rate samples, half a second of observation, and sufficiently stable throughput.
It disappears when throughput varies or the latest sample is stale. It is not a
guaranteed duration or a whole-overview ETA. Phases with unknown work totals show
elapsed time and available counters. Mixed column sizes and high-cardinality
work can remain unpredictable even after an initially stable estimate.

## Cancel or bound elapsed time

```python
token = fw.CancellationToken()
# Another thread, or your progress callback, may call token.cancel().
try:
    overview = fw.explore(df, progress=True, cancel=token, timeout=60)
except fw.AnalysisCancelled:
    print("Analysis stopped")
```

Timeouts are finite nonnegative seconds measured with a monotonic clock.
Cancellation is cooperative: checks occur between work items and fingerprint
chunks. A single pandas/NumPy operation must return before cancellation can take
effect, so this is not a hard process deadline. Cancellation raises
`AnalysisCancelled`; it does not return an apparently complete partial result.
Keyboard interrupts also clean up call resources. Progress and cancellation
objects are never saved in analytical results or recipe parameters. Supply these
controls when calling `Recipe.run`.

## Choose the work you need

```python
overview = fw.explore(
    df,
    sections=["missingness", "dependencies"],
    options={
        "missingness": {"features": ["site", "visit", "value"]},
        "dependencies": {
            "max_key_size": 1,
            "max_candidates": 5,
            "max_dependency_tests": 100,
            "include_grain": False,
        },
    },
    progress=True,
)
```

The four overview sections are `missingness`, `dependencies`, `paths`, and
`value_patterns`. Omitted sections are explicitly `not_requested`; text, HTML,
and SVG presentations identify them. `options` accepts each requested
operation's analytical options, such as `features`, `max_pairs`, or
`max_candidates`. Shared source settings (`scope`, `missing`, `table_id`) and
runtime controls stay on the overview.

Shared parameters (`features`, `by`, entity settings) flow to their relevant
sections; path search settings belong in `options["paths"]`. Use
`options["dependencies"]` to override the overview's default 20 single-column
dependency candidates. Separate calls to
`discover_dependencies` still default to 100 candidates and maximum key size two.

Dependency options available both directly and in `options["dependencies"]`:

| Option | Effect |
| --- | --- |
| `include_grain=False` | Skip foundation grain graphs; retain dependency tests and candidate summaries. |
| `max_grain_views=n` | Build at most n distinct supported population views in their usual order. |
| `max_dependency_tests=n` | Cap discovery candidate/target/context tests in their usual order. |

These options default to the previous complete work within the existing search
budgets. Graph metadata records possible/omitted views and reasons for excluded
candidates. Test coverage records possible/omitted tests. A test budget does not
cap foundation graph computations; combine it with `include_grain=False` or a
graph-view budget to control both. An untested relation is not negative evidence.
Candidate summaries are still computed up to `max_candidates`, including when
`max_dependency_tests=0`.
`min_accuracy` and example/display limits primarily control output, not test work.

## What is reused and what still costs time

One public call owns a private context. Nested overview components share its
source fingerprint, scoped frame, selected value encodings, and presence masks.
The context is discarded after success, failure, or cancellation. A later call
revalidates the source, including mutations. There is no persistent dataframe
cache or public session to invalidate manually.

Fingerprinting hashes the index and every column with vectorized pandas hashing
(about 0.04 s for 500,000 rows × 14 columns); only object columns need a Python
pass to hash a typed representation of each cell. Selecting features or a scope
reduces analytical work but does **not** eliminate full-source fingerprinting. Physically
projecting the dataframe reduces that scan and creates a different source identity;
use the same projected source for subsequent inspection.

Preparation encodes only needed values. Native presence-only columns can use
missing masks without retaining value codes. Row-based availability reuses masks,
entity counts use integer groups, and signatures occupy packed bits. Modal counts
and path prefixes use integer grouping; string patterns count repeated strings
once and weight their frequencies. Bounded examples avoid allocating every match.
`select` evaluates a finding's saved predicate directly rather than replaying
unrelated graphs and summaries; source identity is still validated.

Graph metrics are reused only for identical eligible populations. Retained graph
masks are packed and interned. The subset-metric cache is capped at 512 entries
and 16 MiB of packed mask keys; the path-prefix array cache is capped at 32 MiB.
The shared preparation dictionary cache is capped at 100,000 total identities;
larger dictionaries are released after encoding. Grain reuses integer groups and missing
codes without retaining cell dictionaries.
These are cache budgets, not whole-operation memory limits. Input frames, encoded
columns, output graphs, and source-position lists can still be large. Work grows
with rows, selected features, candidate/target tests, distinct context groups,
and graph views. Unique IDs, continuous values, long strings, and mixed scalar
object columns remain more expensive than repeated categorical values.

## Saved result size

`to_dict()` exports are plain JSON. Findings and grain views keep at most
`example_limit` source positions each, so exports do not grow with row count for
those records. Export size is driven mostly by the number of findings, completed
dependency tests and foundation grain structures; narrow `features`, candidate and
pair budgets on wide frames. (The optional `fieldwork.compact` envelope of 0.1.x
was removed: it saved 5–30% by deduplicating containers that exports no longer
repeat.)

See the [implementation measurements](performance-results.md) for measured
runtime, memory, dataset definitions, and reproduction commands.
