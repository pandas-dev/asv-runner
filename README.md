# pandas ASV Runner

This repository is utilized for running the ASV suite on each commit to the main branch of pandas via GitHub actions.

Features:

- Each hourly run uploads its results as a GitHub Actions artifact (`result-<sha>`, kept 90 days), and the nightly run stores them in Parquet in the Hugging Face dataset [`pandas-dev/asv-runner`](https://huggingface.co/datasets/pandas-dev/asv-runner). Each nightly update is checked to change no stored rows before it is pushed, and history is squashed monthly. Which commits have been claimed and benchmarked is tracked in `data/shas.txt` on the small `asv-claims` branch of this repo.
- Publishes the results on https://pandas-dev.github.io/asv-runner/: a static site built nightly from the stored results, with charts per benchmark, recent regressions and improvements, a view per commit, and markers where a benchmark's code or the environment changed. Links to the site reproduce the exact chart (parameters, zoom, options).
- Uses pandas' own `setup-pixi` GitHub action and `build-pandas` pixi task, in pandas' `asv` pixi environment. Any changes to the pandas build process will not necessitate any maintenance here.
- Stores the environment on each run. The site and the regression issues show which packages changed between benchmarked commits.
- Runs on GitHub actions once every hour, checking if there are new commits to run. In doing so it runs the most recent commit that has no benchmarks, up to 40 commits back. If there are multiple commits within an hour, we will not run older ones immediately. However they will likely be run in the future, unless the average velocity of commits is roughly more than 1 per hour over 2 days.
- Automatically detects regressions. Running the ASV suite on GitHub actions workers introduces a significant amount of noise. As such, the detection of automatic regressions is geared to minimize false positives. It operates by comparing a rolling min of the benchmarks to a rolling max, dubbing these the `established_best` and `established_worst`, and flags a regression when the current `established_best` is 5% higher than a previous `established_worst` (and an improvement in the mirror case). Windows never span a change of a benchmark's code, or a commit listed in `asv_runner/breaks.toml`: commits where how results are measured changed (benchmark tooling, compilers, runner), so results before and after are not comparable. It seems to also not have many false negatives, although this is much harder to establish.
- When a regression is detected, a GitHub issue is opened with the title set to the commit that the regression was detected on. The issue will have a link to the commits between benchmark runs (in case there are commits where benchmarks were skipped), which benchmarks regressed, absolute and relative regression metrics, and links to the ASV charts for each benchmark. By using GitHub issues, the triaging and investigation of flagged regressions can be coordinated by a team of developers. When one commit flags more than 100 benchmarks, a single issue labelled `measurement-change` is opened instead, laying out what changed and how to add the commit to `breaks.toml`.
- All historic ASV results are stored losslessly as Parquet under `data/` in the Hugging Face dataset [`pandas-dev/asv-runner`](https://huggingface.co/datasets/pandas-dev/asv-runner): `runs`, `benchmarks` and `values` (partitioned by week), plus environments, benchmark definitions, and commit titles. They are more convenient to analyze than the raw JSON files, which the store can rebuild exactly, and can be read without downloading the whole dataset, e.g. `pl.scan_parquet("hf://datasets/pandas-dev/asv-runner/data/values/**/*.parquet")`.

## Development

Requires Python 3.14.

```sh
pip install -e ".[dev]"
pytest tests/
prek run --all-files
```

The hourly benchmark run executes `python -m asv_runner` from a source checkout without installing the package, so the modules it uses (`claim`, `benchmark`, `stage_results`, `util`) import only the standard library.
