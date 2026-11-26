# Running on a free GPU (Google Colab or Kaggle)

1. Open https://colab.research.google.com, then go to **Runtime → Change runtime type → T4 GPU**.
2. Run each block below in its own cell.

```python
# Cell 1: install RAPIDS (Colab's guide: https://rapids.ai/start.html, under "Colab")
!pip install -q --extra-index-url=https://pypi.nvidia.com "cudf-cu12" "cuml-cu12" xgboost
!nvidia-smi
```

```python
# Cell 2: get the code
!git clone https://github.com/rafayelmirijanyan1997/rapids-vs-cpu-benchmark.git
%cd rapids-vs-cpu-benchmark
```

```python
# Cell 3: run the benchmark at 1, 3 and 6 months of data on both backends
for months in [["2023-01"], ["2023-01","2023-02","2023-03"],
               ["2023-01","2023-02","2023-03","2023-04","2023-05","2023-06"]]:
    m = " ".join(months)
    !python src/benchmark.py --backend cpu --months {m}
    !python src/benchmark.py --backend gpu --months {m}
```

```python
# Cell 4: build the table and chart, then download the results
!python src/report.py
from google.colab import files
files.download("results/runs.jsonl"); files.download("results/stage_times.png")
```

Copy `runs.jsonl`, `results.md` and `stage_times.png` into `results/` in your local repo and commit them.

If the T4's 15 GB of memory runs out at 6 months, drop to 4 months and say so in the README. The limit is a finding in its own right.

Profiling (for extra credit): run `!nsys profile -o gpu_run python src/benchmark.py --backend gpu --months 2023-01`. Open the report in Nsight Systems and add a screenshot of the timeline.
