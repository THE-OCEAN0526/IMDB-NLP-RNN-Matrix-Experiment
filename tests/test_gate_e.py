"""Gate E: independently recalculate locked one-time official Test, without model inference."""
from __future__ import annotations
import json
import unittest
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,log_loss

from src.common import ROOT,file_hash

R=ROOT/"results"
TEST=R/"final_test"
SELECTED=["ML-04-s42","DL-05-s42"]

class GateE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock=json.loads((R/"model_selection_lock.json").read_text())
        cls.pre=json.loads((R/"final_test_preflight.json").read_text())
        cls.data=json.loads((TEST/"locked_final_test_metrics.json").read_text())
        cls.manifest=json.loads((TEST/"final_test_output_manifest.json").read_text())
        cls.test=pd.read_parquet(ROOT/"data/processed/test.parquet")
        cls.raw=pd.read_parquet(ROOT/"data/raw/test.parquet")
        cls.runs=pd.read_csv(R/"runs.csv")
        cls.pred={rid:pd.read_csv(TEST/f"{rid}_test_predictions.csv") for rid in SELECTED}

    def test_a_one_time_lock_and_fixed_scope(self):
        self.assertEqual(self.pre["status"],"PASS")
        self.assertEqual(self.lock["status"],"LOCKED_BEFORE_FINAL_TEST")
        self.assertEqual(self.lock["future_final_test_scope"],SELECTED)
        self.assertEqual(self.data["scope"],SELECTED)
        self.assertEqual(self.data["model_evaluations_performed"],2)
        self.assertTrue(self.data["selection_was_validation_only"])
        self.assertFalse(self.data["test_based_retuning"])
        self.assertEqual(self.data["model_selection_lock_sha256"],
                         file_hash(R/"model_selection_lock.json"))
        self.assertEqual(self.data["preflight_sha256"],file_hash(R/"final_test_preflight.json"))

    def test_b_preserve_original_official_test_25k(self):
        self.assertEqual(len(self.test),25000)
        self.assertEqual(len(self.raw),25000)
        self.assertEqual(self.test.source_index.tolist(),list(range(25000)))
        self.assertEqual(self.test.label.tolist(),self.raw.label.tolist())
        self.assertEqual(self.test.text.tolist(),self.raw.text.tolist())
        self.assertEqual(self.test.label.value_counts().sort_index().to_dict(),{0:12500,1:12500})
        self.assertEqual(file_hash(ROOT/"data/raw/test.parquet"),self.data["test_source_sha256"])
        self.assertEqual(file_hash(ROOT/"data/processed/test.parquet"),self.data["test_processed_sha256"])

    def test_c_model_hashes_and_original_registry_unchanged(self):
        self.assertEqual(file_hash(R/"runs.csv"),self.lock["runs_registry_sha256"])
        self.assertEqual(len(self.runs),14)
        self.assertTrue(self.runs.run_id.is_unique)
        for rid,record in self.lock["source_artifacts"].items():
            self.assertEqual(file_hash(ROOT/record["model_path"]),record["model_sha256"])
            self.assertEqual(file_hash(ROOT/record["validation_predictions_path"]),record["validation_predictions_sha256"])

    def test_d_validation_registries_still_have_no_test_metrics(self):
        self.assertTrue(self.runs[["test_accuracy","test_precision","test_recall","test_f1","test_loss"]].isna().all().all())
        self.assertTrue((self.runs.status=="validation_complete").all())
        self.assertEqual(self.data["baseline_runs_registry_sha256"],self.lock["runs_registry_sha256"])

    def test_e_both_25k_prediction_files_align_one_to_one(self):
        for rid,p in self.pred.items():
            self.assertEqual(len(p),25000,rid)
            self.assertEqual(p.example_id.tolist(),self.test.example_id.tolist())
            self.assertEqual(p.source_index.tolist(),list(range(25000)))
            self.assertEqual(p.label.tolist(),self.test.label.tolist())
            self.assertEqual(p.example_id.nunique(),25000)
            self.assertTrue(set(p.prediction.unique()).issubset({0,1}))
            self.assertEqual(len(p.columns),5)

    def test_f_recalculate_test_metrics_and_confusion(self):
        for rid,p in self.pred.items():
            stored=self.data["predictions_and_metrics"][rid]
            y=p.label.to_numpy(dtype=int)
            pred=p.prediction.to_numpy(dtype=int)
            expected={
                "accuracy":accuracy_score(y,pred),
                "precision":precision_score(y,pred,zero_division=0),
                "recall":recall_score(y,pred,zero_division=0),
                "f1":f1_score(y,pred,zero_division=0),
            }
            for metric,val in expected.items():
                self.assertAlmostEqual(stored[metric],val,places=12,msg=rid+" "+metric)
            matrix=confusion_matrix(y,pred,labels=[0,1]).tolist()
            self.assertEqual(stored["confusion_matrix"],matrix)
            self.assertEqual(sum(sum(r) for r in matrix),25000)

    def test_g_svc_fixed_decision_rule(self):
        rid=SELECTED[0]
        p=self.pred[rid]
        d=self.data["predictions_and_metrics"][rid]
        self.assertTrue(np.isfinite(p.decision_score.to_numpy()).all())
        self.assertTrue(np.array_equal((p.decision_score.to_numpy()>=0).astype(int),p.prediction.to_numpy()))
        self.assertIsNone(d["test_loss"])
        self.assertEqual(d["model_sha256"],self.lock["source_artifacts"][rid]["model_sha256"])

    def test_h_lstm_fixed_threshold_and_bce(self):
        rid=SELECTED[1]
        p=self.pred[rid]
        d=self.data["predictions_and_metrics"][rid]
        self.assertTrue(p.positive_probability.between(0,1).all())
        self.assertTrue(np.array_equal((p.positive_probability.to_numpy()>=0.5).astype(int),
                                        p.prediction.to_numpy()))
        self.assertAlmostEqual(d["test_loss"],
                               log_loss(p.label,p.positive_probability,labels=[0,1]),places=7)
        self.assertEqual(d["trainable_params"],1242305)
        self.assertEqual(d["model_sha256"],self.lock["source_artifacts"][rid]["model_sha256"])

    def test_i_locked_output_manifest_hashes(self):
        self.assertEqual(self.manifest["model_selection_lock_sha256"],
                         self.pre["model_selection_lock_sha256"])
        self.assertEqual(self.manifest["metrics_sha256"],file_hash(TEST/self.manifest["metrics_file"]))
        for basename,sha in self.manifest["output_files_sha256"].items():
            self.assertEqual(file_hash(TEST/basename),sha,basename)
        for rid,meta in self.data["predictions_and_metrics"].items():
            self.assertEqual(file_hash(TEST/meta["predictions_csv"]),meta["predictions_sha256"])

    def test_j_both_test_confusion_png_and_comparison(self):
        for rid in SELECTED:
            fig=TEST/f"{rid}_test_confusion_matrix.png"
            self.assertTrue(fig.is_file())
            self.assertGreater(fig.stat().st_size,5000)
        chart=TEST/self.data["comparison_figure"]
        self.assertTrue(chart.is_file())
        self.assertGreater(chart.stat().st_size,5000)

    def test_k_pretest_snapshot_preserved_and_final_notebook_executed(self):
        snap=ROOT/"notebooks/07_IMDB_Results_Comparison_PRETEST.ipynb"
        pre=json.loads((R/"phase4_pretest_acceptance.json").read_text())
        self.assertEqual(file_hash(snap),pre["notebook_07"]["sha256"])
        final=ROOT/"notebooks/07_IMDB_Results_Comparison.ipynb"
        book=nbformat.read(final,as_version=4)
        cells=[c for c in book.cells if c.cell_type=="code"]
        self.assertGreaterEqual(len(cells),13)
        self.assertTrue(all(c.execution_count is not None for c in cells))
        errors=[o for c in cells for o in c.get("outputs",[]) if o.get("output_type")=="error"]
        self.assertFalse(errors)
        self.assertTrue(any("Final Test" in c.source or "官方 Test" in c.source for c in book.cells if c.cell_type=="markdown"))

    def test_l_final_summary_and_gate_e_docs_present(self):
        self.assertTrue((R/"GATE_E_REPORT.md").is_file())
        self.assertTrue((R/"gate_e_acceptance.json").is_file())
        self.assertTrue((R/"FINAL_TEST_PROTOCOL.md").is_file())
        self.assertTrue((R/"PHASE4_PRETEST_REPORT.md").is_file())
        self.assertTrue((R/"model_selection_lock.json").is_file())
        self.assertEqual(self.data["status"],"OFFICIAL_TEST_EVALUATED_ONCE")

if __name__=="__main__":
    unittest.main(verbosity=2)
