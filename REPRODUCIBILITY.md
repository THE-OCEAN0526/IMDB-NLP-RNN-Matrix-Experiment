# Reproducibility — data, software and known limitations

## Experiment software and hardware

- Python 3.11.17
- TensorFlow 2.20.0 / Keras 3.15.1
- NumPy 2.2.6, pandas 2.3.3, scipy 1.17.1
- scikit-learn 1.9.1, matplotlib 3.11.2
- pyarrow 26.0.0, joblib 1.6.0, PyYAML 6.0.3
- Linux + JupyterLab; NVIDIA RTX 2060 GPU for the recurrent models; sklearn SVC on CPU.
- GloVe models were trained with GPU acceleration in a separate environment configured with the same software image.

The provided `requirements.txt` records tested-in-place installed versions; a fresh install on an arbitrary platform has **not** been tested.

## Data source (not redistributed)

Original benchmark: https://ai.stanford.edu/~amaas/data/sentiment/

The project expects exactly 25,000 original train and 25,000 official test entries in Parquet, with DataFrame columns [text, label], labels 0/1. The original configuration refers to these local files:

- `/workspace/20261103_NLP_assets/data/train.parquet`
- `/workspace/20261103_NLP_assets/data/test.parquet`

Expected original file SHA256s:
- train: db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c
- test: b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da

Another download or reserialization of the same dataset may produce a **different Parquet byte hash**. Adjusting file paths or source hashes without tracking provenance will not demonstrate exact reproducibility. Prepare an explicit new configuration/version before rerunning with different byte representations. The published results correspond to the saved experimental runs; a clean-environment end-to-end rerun has not been independently verified.

## Notebook order

1. `notebooks/00_IMDB_Data_Preparation.ipynb`
2. `notebooks/01_IMDB_Preprocessing_Vectorization.ipynb`
3. `notebooks/02_IMDB_Traditional_ML_Matrix.ipynb`
4. `notebooks/03_IMDB_RNN_Baselines.ipynb`
5. `notebooks/04_IMDB_LSTM_Regularization_Matrix.ipynb`
6. `notebooks/05_IMDB_Advanced_RNN_Architectures.ipynb`
7. `notebooks/07_IMDB_Results_Comparison.ipynb`
8. `notebooks/06_IMDB_GloVe_Transfer_Learning.ipynb` (post-official-Test exploratory validation)

The code and notebooks often expect exact artifact paths relative to the original project. Download and extract the full research artifact bundle from Releases and supply raw review data locally before attempting full reruns.

Never run `Run All` on historical training notebooks without reviewing their overwrite guards. The official Test results were locked after one selected evaluation for ML-04 and DL-05; do not use Test to retune or reselect.

## Scientific limitations

- Only one seed (42), no multiseed confidence intervals.
- Some hyperparameters and EarlyStopping were chosen using the same Validation split used to compare models, potentially biasing Validation F1 upwards.
- RNNs truncate inputs to 160 tokens; ML vectors represent full text.
- Exclusion checks cover *identical cleaned text*, not near-duplicate semantics.
- GloVe was performed after official Test: exploratory Validation only.
- Do not load untrusted `.joblib` or `.keras` files without checking provenance.
