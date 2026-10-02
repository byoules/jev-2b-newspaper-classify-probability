# Open-Jev news headline evaluation

Classify news headlines as **sports**, **politics**, or **finance** using Open-Jev-2B. The pipeline returns a probability for each category, selects the most likely category, and compares the prediction with an existing label.

The true label is never included in the model input. This project performs inference and evaluation; it does not train a model.

## Requirements

- Linux, or a Linux environment with Bash.
- Python 3.10–3.13 with venv support; Python 3.12 is recommended.
- An NVIDIA GPU and a driver compatible with CUDA 12.8 for GPU execution.
- At least 8 GiB of free memory on one visible GPU by default.
- Internet access for initial package, dataset, and model downloads.
- Several GB of free disk space for dependencies and cached model weights.

The launcher installs PyTorch 2.8.0 with CUDA 12.8, which supports NVIDIA Blackwell GPUs. A locally installed CUDA toolkit is not required for the prebuilt PyTorch wheel. CPU execution is available explicitly, but is slower and requires sufficient system RAM.

## Quick start

Clone the repository:

```bash
git clone https://github.com/byoules/jev-2b-newspaper-classify-probability.git
cd jev-2b-newspaper-classify-probability
```

Alternatively, download and extract the repository ZIP, then open a terminal in the project folder.

Test three headlines:

```bash
bash run.sh --limit 3
```

Run the full dataset:

```bash
bash run.sh
```

The first run creates an isolated `.venv`, installs dependencies, downloads the dataset when missing, checks GPU availability and free memory, and downloads and loads the model. Later runs reuse the environment, dataset, and cached weights. Results are saved in a timestamped folder under `outputs/`.

The pipeline runs as a standalone Python process and releases its GPU allocations when it exits. It does not require a model server, llama.cpp, Ollama, a GUI, or a database.

## Run on a Linux server

Connect to the server through SSH or a remote terminal. Clone the repository there, or copy the project to a server folder you can write to. Run the commands in the server's Linux terminal:

```bash
cd jev-2b-newspaper-classify-probability
bash run.sh --check
bash run.sh --limit 3
bash run.sh
```

The `cd` command assumes you are in the parent folder of the project. If you extracted the ZIP, use the name of its extracted folder instead.

If you access the server's files through a mapped Windows drive, you can copy files and open completed reports from Windows. Launch the pipeline in the server terminal so it uses the server's Python environment and GPU. The Linux launcher does not run in Windows CMD.

On a shared server, the pipeline chooses a visible GPU with enough free memory. It stops if none meets the configured threshold. You can select a specific GPU with `--device cuda:N`. Downloads, dependencies, and outputs stay in the project folder, and other users' running processes are left alone.

## Other commands

```bash
bash run.sh --check          # Show CUDA and GPU diagnostics
bash run.sh --limit 3        # Score the first three headlines
bash run.sh --download-only # Download data and model files without inference
bash run.sh --device cuda:1 # Select a particular visible GPU
bash run.sh --device cpu    # Run explicitly on CPU
```

GPU indices refer to the devices visible to PyTorch. `CUDA_VISIBLE_DEVICES` can remap those indices. Containers must have NVIDIA GPU access enabled.

The default uses one candidate at a time to limit peak GPU memory. The 8 GiB memory threshold is a conservative check, not a guarantee for every input. If CUDA is unavailable or insufficient memory is free, the pipeline stops with an error. It does not stop other processes or modify system packages or drivers.

If PyTorch reports that a Blackwell GPU with capability `sm_120` is unsupported, install the correct wheel in the project environment:

```bash
.venv/bin/python -m pip install --upgrade 'torch==2.8.0+cu128' --index-url https://download.pytorch.org/whl/cu128
```

## Dataset

The default example dataset is publicly available on Kaggle:

- URL: https://www.kaggle.com/datasets/brasks43/news-articles-classified
- Identifier: `brasks43/news-articles-classified`

The version inspected during development contains 30 headlines in `news_classification_dataset.csv`, with these columns:

| Column | Description |
|---|---|
| `id` | Record identifier |
| `headline` | Text to classify |
| `true_label` | Existing label: sports, politics, or finance |

The pipeline discovers the actual CSV filename and validates the required columns. It downloads through Kaggle's official `kagglehub` SDK only when no matching local CSV exists. Updates on Kaggle do not automatically replace the local copy.

### Use your own data

Place a CSV with the required columns in `data/`, or set `local_csv` in `settings.json` to its path. If multiple matching CSV files exist, set `local_csv` explicitly.

The current pipeline expects the three categories sports, politics, and finance. Supporting a different category set requires updating the category validation in `pipeline.py` as well as the definitions in `settings.json`.

Missing headlines produce error rows. Invalid ground-truth labels are reported and excluded from labeled evaluation, although valid headlines can still be scored. Duplicate IDs and headlines are reported and retained, including in metrics. Original headline and label text are preserved; label comparisons strip whitespace and ignore case.

### Kaggle credentials

Private or consent-gated datasets may require authentication. Configure a Kaggle token in `~/.kaggle/access_token`, or legacy credentials in `~/.kaggle/kaggle.json`. Existing supported Kaggle authentication can also be reused.

Alternatively, download the CSV manually and place it in `data/`. Credentials are not included in this repository. The scoring process does not prompt interactively for authentication. A download failure stops execution without generating substitute data.

## Configuration

Edit `settings.json` to change:

- Dataset identifier or explicit local CSV path.
- Data, model-cache, and output folders.
- Device selection and minimum free GPU memory.
- Candidate batch size, maximum input length, and row limit.
- Classification question and category descriptions.

Paths are relative to the project folder unless absolute paths are supplied.

The pipeline uses the checkpoint's saved calibration temperature and disables prefix caching. Inherited JEV quantization, dtype, and device-map overrides are rejected to preserve the expected inference configuration. Inputs that exceed the configured token limit fail rather than being silently truncated.

Changing category descriptions, precision, or checkpoint versions changes the experiment.

## Outputs

Each run creates a separate timestamped folder under `outputs/`.

| File | Contents |
|---|---|
| `scores.csv` | Original records, category probabilities, prediction, confidence, correctness, timing, status, and error |
| `incorrect-predictions.csv` | Successfully scored records that disagree with valid ground-truth labels |
| `report.md` | Row counts, accuracy, per-category precision/recall/F1, confusion matrix, Brier score, log loss, and limitations |
| `metrics.json` | Machine-readable evaluation metrics |
| `data-issues.csv` | Missing values, invalid labels, and duplicate records |
| `raw-responses.jsonl` | Open-Jev responses and inference provenance |
| `run-metadata.json` | Input checksum, dependencies, settings, device, revisions, saved temperature, and run status |
| `error.txt` | Traceback for a model setup or run-level failure, when applicable |

`confidence` is the highest category probability. Scores are flushed after each row. Failed rows have blank prediction and probability fields. A CUDA scoring failure stops the run and preserves partial results. Failures return a nonzero exit code. Setup failures may produce metadata and an error report without a scores file.

## Model and implementation

- Model: https://huggingface.co/ZefanCai/Open-Jev-2B
- Model package revision: `0c7aa498b1627be8da4acf34c863ff0ee0a92785`
- Base model: `Qwen/Qwen3.5-2B`
- Base revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`
- Official implementation: https://github.com/Zefan-Cai/Open-Jev
- Vendored code commit: `820e2a7cacf2ac906b1a5590ee9a1a8afc1d0312`

The repository includes the upstream runtime source and its licenses. Model weights are downloaded separately.

The official `load_predictor` function loads the adapter, scalar decision head, and saved calibration temperature. The official choice interface computes probabilities over the supplied candidates. Probabilities are not produced by asking a text-generation model to write confidence percentages. Checkpoint metadata is checked against the pinned base-model revision before loading.

Open-Jev is an independent open model, separate from TypeSafe's hosted Jev service.

## Evaluation limits

The default dataset's 30 labeled headlines support an exploratory comparison, not a reliable estimate of general performance or probability calibration. High confidence does not guarantee a correct prediction. Labels can be ambiguous, and headlines can cover overlapping topics.

The pipeline does not fit calibration on the evaluation data or claim that the probabilities are calibrated for a new dataset. Inspect individual disagreements alongside the aggregate metrics.

Metric conventions:

- Multiclass Brier score is the mean of the sum of squared errors across all categories.
- Log loss uses natural logarithms, with probabilities clipped below at `1e-15`.
- Precision, recall, and F1 are reported as zero when their denominator is zero.
- Metrics exclude failed rows and invalid ground-truth labels.
- Duplicate rows remain included and are identified in `data-issues.csv`.
- Model loading time and per-headline inference time are recorded separately.

## Tests

Run the software checks without downloading model weights:

```bash
python3 test_pipeline.py
```

These checks cover the official request/response schema, separation of ground-truth labels from inputs, probability validation, known-answer metric calculations, data issue reporting, and CSV selection.

The fixed probability distributions used in software tests are contract fixtures, not model predictions. Use `bash run.sh --limit 3` to test real model inference on your hardware.

## Sources and licenses

- Open-Jev: https://github.com/Zefan-Cai/Open-Jev
- Open-Jev-2B: https://huggingface.co/ZefanCai/Open-Jev-2B
- Kaggle SDK: https://github.com/Kaggle/kagglehub
- PyTorch: https://pytorch.org/get-started/locally/

Original upstream licenses and notices are retained in `vendor/open-jev/`. See `UPSTREAM.txt` for provenance. Dataset and model terms remain those of their publishers.

## Keywords

Open-Jev, Jev, news classification, headlines, sports, politics, finance, probabilities, Kaggle, Qwen, PyTorch, CUDA, evaluation, Brier score, log loss

