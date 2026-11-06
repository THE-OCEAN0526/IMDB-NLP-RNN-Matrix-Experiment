# Data, licenses and distribution

## IMDB reviews
Movie reviews are third-party content. See the [Stanford dataset](https://ai.stanford.edu/~amaas/data/sentiment/) and [IMDb data-use restrictions](https://help.imdb.com/article/imdb/general-information/can-i-use-imdb-data-in-my-software/G5JTRESSHJBBHTGX).

To avoid redistributing third-party original review text, this public repository and Release bundle exclude:
- `data/raw/train.parquet` and `data/raw/test.parquet`
- `data/processed/train.parquet`, `validation.parquet`, `test.parquet`

The non-text provenance, split assignments and derived predictions are included. Users should independently retrieve the dataset and comply with applicable third-party terms.

## GloVe
Pretrained Stanford GloVe vectors are distributed under [Public Domain Dedication and License v1.0](https://opendatacommons.org/licenses/pddl/1.0/). The GloVe source used here was provided as a word2vec-binary representation via the documented Zenodo mirror, *not* the official Stanford plain-text archive. See `optional_glove/results/OPTIONAL_GLOVE_REPORT.md` for provenance and hashes.

## Project code
The owner has authorized a public GitHub repository, **not a particular open-source license**. Unless the owner later supplies a license, do not infer an MIT/Apache or other reuse grant for original project code from public visibility.

Third-party libraries have their own licenses. Never deserialize untrusted Python joblib files.
