# Experiment milestones and validation protocol

This page summarizes the research workflow. Full per-run results, saved model settings, checksums, and validation reports are provided under `results/` and `optional_glove/results/`.

## Dataset preparation — Gate A

- Start from the IMDB official Train (25,000) and Test (25,000) sources.
- Identify 123 Train examples identical to Official Test after text cleaning; exclude them from the training pool.
- Exclude 98 additional duplicate Train examples using the first source index as the keeper.
- Split the remaining 24,779 examples with stratified seed 42: Train 19,823, Validation 4,956.
- Keep Official Test unchanged at 25,000. Validate that exact cleaned-text overlap across Train, Validation and Test is zero.
- Keep the original Test untouched for the later locked evaluation. This is not a semantic near-duplicate test.

## Traditional ML and recurrent baselines — Gate B

- Four traditional ML configurations: BoW / TF-IDF × KNN / linear SVC.
- Three recurrent configurations: SimpleRNN, LSTM, GRU with trainable 100-dimensional Embedding.
- Fit vocabularies and vectorizers on Train only. Select classification hyperparameters using Validation F1.
- Preserve per-run Validation predictions, model artifacts and original plots.

## Regularization and advanced RNNs — Gate C

- Add LSTM Dropout 0.3, EarlyStopping, and their combination, reusing the existing LSTM baseline.
- Compare four advanced models: stacked LSTM, LSTM + GRU, bidirectional LSTM, bidirectional GRU.
- This produces **14 pre-Test experiment conditions** in total.
- EarlyStopping monitors Validation Loss with patience 3 and restores best weights.

## Model comparison and one-time official Test — Gate E

- Before accessing the official Test, rank all 14 conditions using Validation F1.
- Lock one traditional ML candidate (ML-04, TF-IDF + SVC) and one deep-learning candidate (DL-05, LSTM + EarlyStopping).
- Perform official Test evaluation for those two locked models; do not tune on Test.
- Official Test F1: ML-04 88.03%; DL-05 83.15%.
- Preserve the official per-example predictions and confusion matrices.

## Optional GloVe comparison — Gate D

- After the official Test, perform a separate exploratory comparison using GloVe 6B 100d initialized Embedding.
- Add Frozen (TL-01) and Fine-tuning (TL-02) conditions, reusing the existing GRU baseline.
- These two runs **only have exploratory Validation results**, and are not candidates in the earlier official Test selection.
- The full study now contains **16 distinct conditions**.

## Limitations

- A single random seed (42) does not establish statistical significance.
- Using the same Validation split for selection and reporting can introduce optimism.
- The RNN input length was 160 tokens; traditional ML used full-text representations.
- Exact text deduplication does not detect all semantically similar reviews.
