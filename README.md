# pandas ASV Runner

This repository is utilized for running the ASV suite on each commit to the main branch of pandas via GitHub actions.

Features:

- Stores historic ASV results on non-default branches (e.g. `pandas_20250203`). Over time, historic results can become a significant size. By using non-default branches to store the results, we can create a new branch and then completely wipe the old branch from the repository's history.
- Publishes the ASV results on https://pandas-dev.github.io/asv-runner/.
- Uses pandas' `setup-conda` and `build_pandas` GitHub actions directly. Any changes to the pandas build process will not necessitate any maintenance here.
- Stores `environment.yml` on each run. If a performance regression occurs because of a change in dependency, it can be detected by comparing this file to the that of the previous commit.
- Runs on GitHub actions once every hour, checking if there are new commits to run. In doing so it runs the most recent commit that has no benchmarks, up to 40 commits back. If there are multiple commits within an hour, we will not run older ones immediately. However they will likely be run in the future, unless the average velocity of commits is roughly more than 1 per hour over 2 days.
- Automatically detects regressions. Running the ASV suite on GitHub actions workers introduces a significant amount of noise. As such, the detection of automatic regressions is geared to minimize false positives. It operates by comparing a rolling min of the benchmarks to a rolling max, dubbing these the `established_best` and `established_worst`, and flags a regression when the current `established_best` is 5% higher than a previous `established_worst`. It seems to also not have many false negatives, although this is much harder to establish.
- When a regression is detected, a GitHub issue is opened with the title set to the commit that the regression was detected on. The issue will have a link to the commits between benchmark runs (in case there are commits where benchmarks were skipped), which benchmarks regressed, absolute and relative regression metrics, and links to the ASV charts for each benchmark. By using GitHub issues, the triaging and investigation of flagged regressions can be coordinated by a team of developers.
- A summary dataset of all historic ASV results is stored in `asv-runner/data/results.parquet`. This contains the metrics for every benchmark on every commit. It is more convenient to analyze than the raw `asv-runner/data/results/` JSON files.
