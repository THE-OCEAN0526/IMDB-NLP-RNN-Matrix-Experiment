"""Gate A acceptance for the approved dedup_v1.1 variant."""
from __future__ import annotations
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.common import ROOT, clean_review, config, file_hash, ids_hash, review_hash, load_splits


class GateA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.cfg_hash=config()
        cls.frames=load_splits()
        cls.meta=json.loads((ROOT/"data/processed/split_manifest.json").read_text())
        cls.audit=json.loads((ROOT/"results/data_audit.json").read_text())
        cls.feat=json.loads((ROOT/"results/feature_metadata.json").read_text())
        cls.original={n:pd.read_parquet(ROOT/f"data/raw/{n}.parquet") for n in ("train","test")}

    def test_a_source_sha_and_version(self):
        self.assertEqual(self.cfg["version"],"dedup_v1.1")
        self.assertEqual(self.meta["version"],"dedup_v1.1")
        self.assertEqual(self.meta["config_sha256"],self.cfg_hash)
        self.assertEqual(self.feat["config_sha256"],self.cfg_hash)
        for n in ("train","test"):
            expected=self.cfg["dataset"][f"sha256_{n}"]
            self.assertEqual(file_hash(Path(self.cfg["dataset"][f"source_{n}"])),expected)
            self.assertEqual(file_hash(ROOT/f"data/raw/{n}.parquet"),expected)

    def test_b_counts_and_balance(self):
        exp={"train":{0:9869,1:9954},"validation":{0:2467,1:2489},"test":{0:12500,1:12500}}
        for n,df in self.frames.items():
            self.assertEqual(len(df),self.cfg["expected_v1_1"][n])
            self.assertEqual(df.label.value_counts().sort_index().to_dict(),exp[n])
            self.assertTrue(df.example_id.is_unique)

    def test_c_official_test_original_and_order(self):
        x=self.frames["test"]
        original=self.original["test"]
        self.assertEqual(x.source_index.tolist(),list(range(25000)))
        self.assertEqual(x.text.tolist(),original.text.tolist())
        self.assertEqual(x.label.tolist(),original.label.tolist())
        self.assertEqual(x.cleaned_text.tolist(),[clean_review(s) for s in original.text])
        self.assertEqual(len(x),25000)
        self.assertEqual(self.audit["test_internal_extra_duplicate_rows"],201)

    def test_d_reproducible_ids_seed42(self):
        original=self.original["train"]
        pool=pd.DataFrame({
            "source_index":np.arange(25000),
            "example_id":[f"imdb-train-{x:05d}" for x in range(25000)],
            "label":original.label.to_numpy(),
            "hash":[review_hash(clean_review(x)) for x in original.text.tolist()],
        })
        test_h={review_hash(clean_review(x)) for x in self.original["test"].text.tolist()}
        eligible=pool.loc[~pool.hash.isin(test_h)]
        unique=eligible.drop_duplicates("hash",keep="first").reset_index(drop=True)
        tr_idx,val_idx=train_test_split(np.arange(len(unique)),test_size=.20,
                random_state=42,stratify=unique.label.to_numpy())
        self.assertEqual(self.frames["train"].example_id.tolist(),unique.iloc[np.sort(tr_idx)].example_id.tolist())
        self.assertEqual(self.frames["validation"].example_id.tolist(),unique.iloc[np.sort(val_idx)].example_id.tolist())
        self.assertEqual(ids_hash(self.frames["train"].example_id),self.cfg["expected_v1_1"]["train_ids_sha256"])
        self.assertEqual(ids_hash(self.frames["validation"].example_id),self.cfg["expected_v1_1"]["validation_ids_sha256"])

    def test_e_exclusions_and_complete_accounting(self):
        p=ROOT/"data/processed"
        a=pd.read_csv(p/"excluded_test_overlap.csv")
        b=pd.read_csv(p/"excluded_intra_train_duplicates.csv")
        self.assertEqual(len(a),123)
        self.assertEqual(len(b),98)
        self.assertEqual(ids_hash(a.example_id),self.cfg["expected_v1_1"]["excluded_test_overlap_ids_sha256"])
        self.assertEqual(ids_hash(b.example_id),self.cfg["expected_v1_1"]["excluded_internal_ids_sha256"])
        self.assertEqual(a.label.value_counts().sort_index().to_dict(),{0:95,1:28})
        self.assertEqual(b.label.value_counts().sort_index().to_dict(),{0:69,1:29})
        retained=pd.read_csv(p/"split_assignments.csv")
        self.assertEqual(len(retained),49779)
        ids=pd.concat([retained.example_id,a.example_id,b.example_id],ignore_index=True)
        self.assertEqual(len(ids),50000)
        self.assertEqual(ids.nunique(),50000)
        self.assertEqual(file_hash(p/"split_assignments.csv"),self.meta["split_assignments_sha256"])

    def test_f_no_cross_split_identical_full_cleaned_text(self):
        hashes={n:set(f.cleaned_hash) for n,f in self.frames.items()}
        self.assertFalse(hashes["train"]&hashes["validation"])
        self.assertFalse(hashes["train"]&hashes["test"])
        self.assertFalse(hashes["validation"]&hashes["test"])
        self.assertEqual(self.audit["train_val_test_cross_split_cleaned_text_overlaps"],
               {"train_validation":0,"train_test":0,"validation_test":0})
        for n,frame in self.frames.items():
            self.assertEqual(frame.cleaned_hash.tolist(),[review_hash(x) for x in frame.cleaned_text])
            self.assertEqual(frame.cleaned_text.tolist(),[clean_review(x) for x in frame.text])

    def test_g_no_global_label_conflict(self):
        df=pd.concat([self.frames[n][["cleaned_hash","label"]] for n in ("train","validation","test")],ignore_index=True)
        self.assertEqual(int(df.groupby("cleaned_hash").label.nunique().gt(1).sum()),0)
        self.assertEqual(self.audit["global_conflicting_label_hash_groups"],0)

    def test_h_data_provenance_hashes(self):
        for n,frame in self.frames.items():
            self.assertEqual(file_hash(ROOT/f"data/processed/{n}.parquet"),self.meta["counts"][n]["sha256"])
            self.assertEqual(ids_hash(frame.example_id.tolist()),self.meta["counts"][n]["source_ids_sha256"])
        self.assertEqual(self.feat["manifest_split_sha256"],self.meta["split_assignments_sha256"])
        self.assertEqual(self.feat["preprocessing_version"],"phase0-text-v1")

    def test_i_bow_tfidf_train_only_and_sparse_files(self):
        p=ROOT/"artifacts/vectorizers"
        tr=self.frames["train"].cleaned_text.tolist()
        ml=self.cfg["traditional"]
        opts={
            "max_features":ml["max_features"],"ngram_range":tuple(ml["ngram_range"]),
            "min_df":ml["min_df"],"lowercase":False,"token_pattern":ml["token_pattern"],
            "stop_words":None,
        }
        re_bow=CountVectorizer(**opts,dtype=np.int32).fit(tr)
        re_tfidf=TfidfVectorizer(**opts,dtype=np.float64,
                       norm=ml["tfidf_norm"],use_idf=ml["use_idf"]).fit(tr)
        for name,expected in (("bow",re_bow),("tfidf",re_tfidf)):
            saved=joblib.load(p/f"{name if name=='tfidf' else 'bow_count'}_vectorizer.joblib")
            self.assertDictEqual(saved.vocabulary_,expected.vocabulary_)
            if name=="tfidf":
                self.assertTrue(np.allclose(saved.idf_,expected.idf_))
            for split,frame in self.frames.items():
                sparse=sp.load_npz(p/f"{name}_{split}.npz")
                self.assertEqual(sparse.shape,(len(frame),12000))
                self.assertEqual(file_hash(p/f"{name}_{split}.npz"),self.feat["sparse_matrices"][f"{name}_{split}"]["sha256"])
                # CSR internal index order can differ between fit_transform and transform.
                # Compare values by (row, column), allowing only IEEE-754 rounding noise.
                expected_matrix=saved.transform(frame.cleaned_text.tolist())
                self.assertEqual(sparse.nnz,expected_matrix.nnz)
                delta=(sparse-expected_matrix).tocsr()
                if delta.nnz:
                    self.assertLess(float(np.max(np.abs(delta.data))),1e-12)
        self.assertEqual(self.feat["fitted_on"],"dedup_v1.1 train only")

    def test_j_keras_tokens_vocab_and_split_alignment(self):
        p=ROOT/"artifacts/vectorizers"
        vocab=(p/"keras_vocabulary_12000.txt").read_text().splitlines()
        self.assertEqual(len(vocab),12000)
        self.assertEqual(vocab[:2],["","[UNK]"])
        self.assertEqual(file_hash(p/"keras_vocabulary_12000.txt"),self.feat["keras_vocabulary_sha256"])
        with np.load(p/"keras_token_ids_12000_160.npz",allow_pickle=False) as arr:
            for split,frame in self.frames.items():
                self.assertEqual(arr[f"X_{split}"].shape,(len(frame),160))
                self.assertEqual(arr[f"ids_{split}"].tolist(),frame.example_id.tolist())
                self.assertTrue(np.array_equal(arr[f"y_{split}"],frame.label.to_numpy()))
                self.assertTrue((arr[f"X_{split}"]>=0).all())
                self.assertTrue((arr[f"X_{split}"]<len(vocab)).all())
        self.assertEqual(file_hash(p/"keras_token_ids_12000_160.npz"),self.feat["keras_token_ids_sha256"])

    def test_k_legacy_v1_0_frozen(self):
        legacy=ROOT.parents[1]  # /workspace/imdb_matrix_experiment
        expected={
          "config/experiment.yaml":"46bbd9516a471e8e08163d02bed3833a0a8ca9982801cbf96e44893554f66e07",
          "data/processed/split_manifest.json":"60373e2185ae57e9ac279b46556c90183d01af3611982c7ff2de79c4af678243",
          "notebooks/00_IMDB_Data_Preparation.ipynb":"05466e515a665ce87ec9a8d3830a35ab32d974c658bb351368d7c0b5765b04fd",
          "notebooks/01_IMDB_Preprocessing_Vectorization.ipynb":"556c58bcc7f1ab02272c647de6bd34673b1c51fa10e8cf611cff1f32395faa0e",
        }
        for name,sha in expected.items():
            self.assertEqual(file_hash(legacy/name),sha,name)


if __name__=="__main__":
    unittest.main(verbosity=2)
