# CPU vs GPU: end-to-end analytics on NYC taxi data

The same data-science pipeline runs on two stacks, and every stage is timed:

| Stage | CPU stack | GPU stack (NVIDIA RAPIDS) |
|---|---|---|
| Load parquet | pandas | cuDF |
| Clean, feature engineering, group-by, joins | pandas | cuDF |
| Train/test split, metrics | scikit-learn | cuML |
| Model | XGBoost `device="cpu"` | XGBoost `device="cuda"` |

**Task:** predict whether a card-paid yellow-cab trip gets a tip of at least 20% of the fare.
**Data:** [NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), 2023, at 1, 3 and 6 months (TODO: fill in the row counts).

## Results

> TODO: paste `results/results.md` here after running. Report only numbers you measured.

![stage times](results/stage_times.png)

**Hardware:** TODO (for example Colab T4 16 GB vs 2-vCPU Xeon; get this from `runs.jsonl`).

## Findings
- TODO: which stages sped up most, and why (group-bys and joins are memory-bound and parallel).
- TODO: whether ROC-AUC matched between backends (it should, within noise).
- TODO: where the GPU did *not* help (small data, transfer overhead), and where GPU memory became the limit.

## Reproduce
```bash
pip install -r requirements.txt           # CPU
python src/benchmark.py --backend cpu --months 2023-01 2023-02 2023-03
python src/report.py
```
For the GPU backend, see [notebooks/run_on_colab.md](notebooks/run_on_colab.md) (free T4).

## Related
Part 1, writing the kernels by hand (naive CUDA, tiled shared-memory and cuBLAS matrix multiply): [Image-processing-kernels-with-CUDA-CPU-and-GPU-](https://github.com/rafayelmirijanyan1997/Image-processing-kernels-with-CUDA-CPU-and-GPU-)

## Author
Rafayel Mirijanyan
