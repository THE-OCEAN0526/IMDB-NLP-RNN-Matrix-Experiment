"""Optional Gate D, a POST-GATE-E exploratory Validation extension only."""
from __future__ import annotations
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import nbformat

from src.common import ROOT,file_hash

OPT=ROOT/"optional_glove"
R=OPT/"results"
IDS=["TL-01-s42","TL-02-s42"]

class OptionalGateD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=yaml.safe_load((OPT/"experiment.yaml").read_text())
        cls.s=json.loads((R/"glove_source_manifest.json").read_text())
        cls.v=json.loads((R/"glove_coverage.json").read_text())
        cls.frozen=json.loads((R/"FROZEN_OFFICIAL_GATE_E_BEFORE_OPTIONAL.json").read_text())
        cls.runs={rid:json.loads((R/f"{rid}.json").read_text()) for rid in IDS}
        cls.matrix=np.load(ROOT/cls.s["aligned_matrix_path"],allow_pickle=False)
        cls.mask=np.load(ROOT/cls.s["aligned_mask_path"],allow_pickle=False)
        with np.load(ROOT/"artifacts/vectorizers/keras_token_ids_12000_160.npz",allow_pickle=False) as a:
            cls.ids=a["ids_validation"]
            cls.y=a["y_validation"]
            cls.X=a["X_validation"]

    def test_a_optional_after_original_gate_e_preserved(self):
        self.assertEqual(self.frozen["original_gate_e"],"PASS")
        self.assertEqual(len(self.frozen["source_artifacts_sha256"]),26)
        self.assertEqual(self.frozen["official_14_runs"],14)
        for path,sha in self.frozen["source_artifacts_sha256"].items():
            self.assertEqual(file_hash(ROOT/path),sha,path)
        self.assertEqual(json.loads((ROOT/"results/gate_e_acceptance.json").read_text())["status"],"PASS")

    def test_b_config_and_two_new_ids_and_original_baseline(self):
        self.assertEqual(self.c["version"],"optional-glove-v1.0")
        self.assertEqual(self.c["new_runs"],IDS)
        self.assertEqual(self.c["baseline_run"],"DL-03-s42")
        self.assertEqual(self.c["comparison"]["official_Test_evaluation"],"prohibited")
        self.assertEqual(self.c["test_policy"],"Test_already_seen_in_prior_work_do_not_reevaluate_or_tune")
        original=pd.read_csv(ROOT/"results/runs.csv")
        self.assertEqual(len(original),14)
        self.assertEqual((original.run_id=="DL-03-s42").sum(),1)
        self.assertFalse(original.run_id.str.startswith("TL").any())

    def test_c_source_binary_hash_400k_100d(self):
        self.assertEqual(self.s["source_format"],"word2vec_binary_float32")
        self.assertEqual(self.s["source_word_count"],400000)
        self.assertEqual(self.s["glove_binary_size_bytes"],163356485)
        self.assertEqual(file_hash(ROOT/self.s["zip_path"]),self.s["zip_sha256"])
        self.assertEqual(file_hash(ROOT/self.s["glove_binary_path"]),self.s["glove_binary_sha256"])
        self.assertEqual(file_hash(ROOT/self.s["aligned_matrix_path"]),self.s["aligned_matrix_sha256"])
        self.assertEqual(file_hash(ROOT/self.s["aligned_mask_path"]),self.s["aligned_mask_sha256"])
        self.assertEqual(file_hash(OPT/"experiment.yaml"),self.s["experiment_config_sha256"])

    def test_d_vocab_pad_oov_and_matched_coverage(self):
        vocab=(ROOT/self.s["vocabulary_path"]).read_text(encoding="utf8").splitlines()
        self.assertEqual(vocab[:2],["","[UNK]"])
        self.assertEqual(len(vocab),12000)
        self.assertEqual(self.matrix.shape,(12000,100))
        self.assertEqual(len(self.mask),12000)
        self.assertFalse(self.mask[0])
        self.assertFalse(self.mask[1])
        self.assertTrue(np.all(self.matrix[0]==0))
        self.assertTrue(np.isfinite(self.matrix).all())
        self.assertEqual(int(self.mask.sum()),self.v["exact_matching_vocab"])
        self.assertAlmostEqual(self.v["vocab_exact_match_fraction"],self.mask[2:].mean(),places=12)
        self.assertGreater(self.v["train_coverage_excluding_UNK"],0.8)
        self.assertTrue(self.v["fit_from_Train_only"])
        self.assertFalse(self.v["validation_or_test_statistics_used"])

    def test_e_train_only_frozen_token_vocab_and_sizes(self):
        self.assertEqual(self.X.shape,(4956,160))
        self.assertEqual(len(self.y),4956)
        self.assertEqual(len(set(self.ids.tolist())),4956)
        self.assertEqual(int(np.sum(self.y)),2489)
        self.assertEqual(self.c["common_training"]["seed"],42)
        self.assertEqual(self.c["common_training"]["embedding_dim"],100)
        self.assertEqual(self.c["common_training"]["hidden_units"],64)
        self.assertEqual(self.c["common_training"]["max_epochs"],10)

    def test_f_both_full_runs_and_files(self):
        for rid,rec in self.runs.items():
            self.assertEqual(rec["run_id"],rid)
            self.assertEqual(rec["dataset"],"dedup_v1.1")
            self.assertEqual(rec["test_evaluations"],0)
            self.assertEqual(rec["validation_size"],4956)
            self.assertEqual(rec["epochs_completed"],10)
            self.assertFalse(rec["dropout"])
            self.assertFalse(rec["early_stopping"])
            self.assertEqual(rec["glove_source_sha256"],self.s["glove_binary_sha256"])
            self.assertEqual(rec["glove_initial_weights_sha256"],self.s["aligned_matrix_sha256"])
            self.assertTrue((ROOT/rec["model_path"]).is_file())
            self.assertTrue((ROOT/rec["predictions_path"]).is_file())
            self.assertEqual(file_hash(ROOT/rec["model_path"]),rec["model_sha256"])
            self.assertEqual(file_hash(ROOT/rec["predictions_path"]),rec["predictions_sha256"])
            self.assertEqual(file_hash(ROOT/rec["history_path"]),rec["history_sha256"])

    def test_g_validation_preds_align_and_metrics_recomputed(self):
        for rid,rec in self.runs.items():
            p=pd.read_csv(ROOT/rec["predictions_path"])
            self.assertEqual(len(p),4956)
            self.assertEqual(p.example_id.tolist(),self.ids.tolist())
            self.assertEqual(p.label.tolist(),self.y.tolist())
            self.assertEqual(p.example_id.nunique(),4956)
            self.assertTrue(p.positive_probability.between(0,1).all())
            self.assertTrue(np.array_equal((p.positive_probability>=0.5).astype(int),p.prediction))
            checks={
              "val_accuracy":accuracy_score(p.label,p.prediction),
              "val_precision":precision_score(p.label,p.prediction),
              "val_recall":recall_score(p.label,p.prediction),
              "val_f1":f1_score(p.label,p.prediction),
            }
            for name,expected in checks.items():
                self.assertAlmostEqual(float(rec[name]),float(expected),places=12,msg=f"{rid}:{name}")
            self.assertEqual(rec["val_confusion_matrix"],
                             confusion_matrix(p.label,p.prediction,labels=[0,1]).tolist())

    def test_h_models_have_same_gru_and_respective_frozen_trainable(self):
        for rid,rec in self.runs.items():
            model=tf.keras.models.load_model(ROOT/rec["model_path"])
            try:
                emb=model.get_layer("embedding")
                gru=model.get_layer("gru")
                self.assertIsInstance(gru,tf.keras.layers.GRU)
                self.assertEqual(gru.units,64)
                self.assertEqual(emb.output_dim,100)
                self.assertEqual(emb.input_dim,12000)
                self.assertTrue(emb.mask_zero)
                self.assertIsInstance(model.layers[-1],tf.keras.layers.Dense)
                should_train=rid=="TL-02-s42"
                self.assertEqual(emb.trainable,should_train)
                self.assertEqual(rec["embedding_trainable"],should_train)
                self.assertEqual(model.count_params(),1231937)
                self.assertEqual(int(sum(np.prod(w.shape) for w in model.trainable_weights)),
                                 1231937 if should_train else 31937)
                final=emb.get_weights()[0]
                if should_train:
                    self.assertGreater(np.linalg.norm(final-self.matrix),0.1)
                else:
                    self.assertTrue(np.array_equal(final,self.matrix))
            finally:
                del model
                tf.keras.backend.clear_session()

    def test_i_loss_history_and_figures(self):
        for rid,r in self.runs.items():
            h=pd.read_csv(ROOT/r["history_path"])
            self.assertEqual(len(h),10)
            self.assertAlmostEqual(h.val_loss.iloc[-1],r["val_loss"],places=7)
            for suffix in ("_loss.png","_val_confusion_matrix.png"):
                p=R/(rid+suffix)
                self.assertTrue(p.is_file())
                self.assertGreater(p.stat().st_size,5000)
        for suffix in ("glove_3way_validation_f1.png","glove_3way_validation_accuracy.png"):
            self.assertTrue((R/suffix).is_file())

    def test_j_comparison_3_conditions_and_independent_results(self):
        p=R/"three_gru_validation_comparison.csv"
        result=pd.read_csv(p)
        self.assertEqual(len(result),3)
        self.assertEqual(result.run_id.tolist(),["DL-03-s42"]+IDS)
        baseline=pd.read_csv(ROOT/"results/runs.csv").set_index("run_id").loc["DL-03-s42"]
        self.assertAlmostEqual(result.iloc[0].val_f1,baseline.val_f1,places=10)
        for rid in IDS:
            self.assertAlmostEqual(
                result.set_index("run_id").loc[rid,"val_f1"],
                self.runs[rid]["val_f1"],places=10)

    def test_k_notebook06_all_cells_executed_without_errors(self):
        p=ROOT/"notebooks/06_IMDB_GloVe_Transfer_Learning.ipynb"
        book=nbformat.read(p,as_version=4)
        code=[cell for cell in book.cells if cell.cell_type=="code"]
        self.assertGreaterEqual(len(code),6)
        self.assertTrue(all(c.execution_count is not None for c in code))
        self.assertFalse([o for c in code for o in c.get("outputs",[]) if o.get("output_type")=="error"])

    def test_l_no_optional_test_predictions_or_official_test_changes(self):
        self.assertFalse(list(OPT.rglob("*test_predictions*")))
        self.assertFalse((OPT/"results/final_test").exists())
        for rid in IDS:
            self.assertEqual(self.runs[rid]["test_evaluations"],0)
        official=pd.read_csv(ROOT/"results/runs.csv")
        self.assertTrue(official[["test_accuracy","test_precision","test_recall","test_f1","test_loss"]].isna().all().all())

if __name__=="__main__":
    unittest.main(verbosity=2)
