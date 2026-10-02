# Open-Jev news headline evaluation

**Person Responsible:** Brad Youles  
**Status:** Active  
**Last Reviewed:** October 2, 2026

## Purpose

Run the actual Open-Jev-2B decision checkpoint on Brad's Kaggle news headlines. Save a probability for sports, politics, and finance, the predicted category, and a comparison with the existing label. The true label is never given to the model. This project performs inference and evaluation; it does not train a model.

## Run on IRISAI

Extract the ZIP into a new project folder on IRISAI. From that folder run:

```bash
bash run.sh
```

First run creates an isolated `.venv`, installs CUDA-enabled PyTorch and the compatible Open-Jev dependencies, downloads the Kaggle data, checks CUDA and available GPU memory, and downloads/loads the model. Subsequent runs reuse the environment, data, and cached weights. Model inference ends when the script finishes, releasing its GPU memory. There is no model server, llama.cpp, Ollama, GUI, or database.

Use a new folder to avoid inheriting an unrelated Python environment. Python 3.12 is recommended; Python 3.10–3.13 with venv support is accepted. Package installation and initial downloads require internet access and several GB of free disk space. The CUDA 12.6 PyTorch wheel requires a compatible NVIDIA driver. No system Python packages, drivers, or existing services are modified.

Useful commands:

```bash
bash run.sh --check          # CUDA and GPU diagnostics, no model download
bash run.sh --limit 3        # Smoke test on three actual headlines
bash run.sh --download-only # Cache data, model adapter/head, and pinned Qwen weights
bash run.sh --device cuda:1 # Select a different visible GPU
```

GPU indices refer to PyTorch's visible devices; CUDA_VISIBLE_DEVICES can remap host indices. If using a container, it must have NVIDIA GPU access. The default requires 8 GiB free on one GPU and uses one candidate at a time to limit peak memory. This is a conservative preflight threshold, not a guarantee for every input. If insufficient memory is available, the pipeline stops and explains why. It never stops OSS 120B, Qwen, or any other running service. CPU execution is available explicitly with `bash run.sh --device cpu`; it is much slower and needs sufficient system RAM.

## Data and credentials

Dataset: https://www.kaggle.com/datasets/brasks43/news-articles-classified  
Kaggle identifier: `brasks43/news-articles-classified`

The published dataset inspected during development contains `news_classification_dataset.csv`, 30 rows, and columns `id`, `headline`, `true_label`. The pipeline discovers the actual filename and validates the columns. It downloads through Kaggle's official kagglehub SDK only when no matching local CSV exists. Updates on Kaggle do not automatically replace your local data.

For private or consent-gated downloads, configure your existing Kaggle credentials. Current token file: `~/.kaggle/access_token`; legacy credentials: `~/.kaggle/kaggle.json`. You can also place the downloaded CSV directly in `data/`. No credentials are included in this project. Authentication is not prompted interactively inside the scoring process. A download failure stops execution without substitute data.

If more than one matching CSV exists, set `local_csv` in settings.json. Missing headlines produce error rows. Invalid labels are reported and excluded from labeled evaluation, although valid headlines can still be scored. Duplicate IDs and headlines are reported and retained, including in metrics. Original headline and label text are preserved; label comparisons strip whitespace and ignore case.

## Settings

Edit settings.json to change local folders, explicit CSV path, device, memory threshold, candidate batch size, maximum token length, row limit, question, or category descriptions. Paths are relative to this project unless absolute paths are supplied. Default categories are sports, politics, finance.

To preserve the published inference identity, the pipeline rejects inherited JEV quantization, dtype, or device-map overrides. Prefix caching is disabled. Inputs longer than the configured limit fail visibly rather than being silently truncated. Changing category definitions, precision, or checkpoint versions changes the experiment.

## Outputs

Each run gets its own timestamped folder under outputs/:

- scores.csv: original records, per-category probabilities, prediction, maximum probability (confidence), correctness, timing, status, and error.
- incorrect-predictions.csv: successfully scored headlines whose predictions disagree with valid true labels.
- report.md: row counts, accuracy, per-category precision/recall/F1, confusion matrix, Brier score, log loss, and limitations.
- metrics.json: machine-readable evaluation metrics.
- data-issues.csv: invalid/missing values and duplicate records; no silent dropping.
- raw-responses.jsonl: real Open-Jev responses and inference provenance.
- run-metadata.json: input checksum, dependencies, prompt/settings, device, source/model revisions, saved temperature, and completion/failure status.
- error.txt: traceback for model setup or run-level failure, when applicable.

Scores are flushed after each row. Row failures leave prediction/probability fields blank. A CUDA failure stops the run and preserves partial results. The exit code is nonzero for failures. Setup failures may produce metadata and an error report without a scores file.

## Model and implementation

Model: https://huggingface.co/ZefanCai/Open-Jev-2B  
Model package revision: `0c7aa498b1627be8da4acf34c863ff0ee0a92785`  
Pinned base: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`  
Official loader: https://github.com/Zefan-Cai/Open-Jev  
Vendored code commit: `820e2a7cacf2ac906b1a5590ee9a1a8afc1d0312`

The archive includes the upstream runtime source and its licenses so no Git installation or runtime repository cloning is needed. Model weights are downloaded separately. The official load_predictor function loads the adapter, scalar decision head, and saved calibration temperature. Choice probabilities are computed through the official inference path; they are not generated confidence percentages. Exact checkpoint metadata is checked before loading the base model. Open-Jev is an independent open model, not TypeSafe's hosted Jev service.

## Evaluation limits

Thirty labeled headlines support an exploratory comparison, not a reliable estimate of broad performance or calibration. A high probability does not guarantee a correct label. The labels may contain ambiguity, and headlines can cover overlapping topics. No calibration is fitted on this evaluation set and no broad calibration claim is made. Inspect disagreements and probabilities alongside the aggregate metrics.

Multiclass Brier is the mean of the sum of squared errors across all classes. Log loss uses natural logarithms with probabilities clipped below at 1e-15. Precision, recall, and F1 use zero for zero denominators. Metrics exclude failed rows and invalid ground-truth labels; duplicate rows remain included. Model load time and per-headline inference time are recorded separately.

## Validation performed before delivery

- Downloaded the real public Kaggle ZIP and inspected its CSV schema, labels, and 30 rows.
- Downloaded and validated the pinned adapter/head checkpoint and verified its Qwen base revision; cached checkpoint reuse passed with internet access disabled.
- Confirmed the official model loader imports with Transformers 5.10.2 and PEFT 0.19.1.
- Downloaded the dataset through Kaggle's SDK and verified local reuse.
- Verified that a run without CUDA records the failure and creates no substitute predictions.
- Checked Python compilation and Linux launcher syntax.
- Tested the real official request/response schema, separation of true labels from inputs, invalid-probability rejection, known-answer metric calculations, duplicate/missing-data reporting, and multiple-CSV handling.
- GPU model inference has not been validated here. Run `bash run.sh --limit 3` on IRISAI for the actual GPU smoke test.

Software contract fixtures in test_pipeline.py are explicitly not model predictions. No fabricated inference results are supplied.

## Sources and licenses

Kaggle SDK: https://github.com/Kaggle/kagglehub  
Original Open-Jev licenses and upstream notices are retained in vendor/open-jev/. See UPSTREAM.txt for provenance. Dataset and model terms remain those of their publishers.

## Keywords

Open-Jev, Jev, news classification, headlines, sports, politics, finance, probabilities, Kaggle, Qwen, PyTorch, CUDA, IRISAI, evaluation, Brier score, log loss
