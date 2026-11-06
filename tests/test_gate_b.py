"""Gate B: all seven full-data baseline conditions have traceable Validation metrics."""
from __future__ import annotations
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

from src.common import ROOT,file_hash
from src.evaluation import phase1_settings

RUNS=["ML-01-s42","ML-02-s42","ML-03-s42","ML-04-s42",
      "DL-01-s42","DL-02-s42","DL-03-s42"]

class GateB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.hash,cls.manifest=phase1_settings()
        cls.rows=pd.read_csv(ROOT/"results/runs.csv")
        cls.val=pd.read_parquet(ROOT/"data/processed/validation.parquet")
        cls.id_to_label=dict(zip(cls.val.example_id,cls.val.label))

    def get(self,run_id):
        rows=self.rows[self.rows.run_id==run_id]
        self.assertEqual(len(rows),1,run_id)
        return rows.iloc[0]

    def test_a_seven_conditions_and_unqiue_runs(self):
        # Gate B is a historical checkpoint: later phases legitimately append runs.
        # Assert the original seven still exist once, not that the registry stays at 7 forever.
        self.assertTrue(set(RUNS).issubset(set(self.rows.run_id)))
        self.assertTrue(self.rows.run_id.is_unique)
        baseline=self.rows[self.rows.run_id.isin(RUNS)]
        self.assertEqual(len(baseline),7)
        self.assertTrue((baseline.status=="validation_complete").all())

    def test_b_gate_a_provenance_and_frozen_settings(self):
        self.assertEqual(self.cfg["dataset_variant"],"dedup_v1.1")
        self.assertEqual(self.cfg["test_policy"],"do_not_evaluate_in_phase1")
        for _,r in self.rows.iterrows():
            self.assertEqual(r.dataset_version,"dedup_v1.1")
            self.assertEqual(r.phase1_config_hash,self.hash)
            self.assertEqual(r.phase0_config_hash,self.manifest["config_sha256"])
            self.assertEqual(r.split_hash,self.manifest["split_assignments_sha256"])
            self.assertEqual(r.preprocess_version,"phase0-text-v1")
            self.assertEqual(int(r.seed),42)

    def test_c_official_test_remains_sealed(self):
        for field in ("test_accuracy","test_precision","test_recall","test_f1","test_loss"):
            self.assertTrue(self.rows[field].isna().all(),field)

    def test_d_all_validation_predictions_and_labels(self):
        for rid in RUNS:
            r=self.get(rid)
            path=ROOT/r.predictions_path
            self.assertTrue(path.is_file(),rid)
            pred=pd.read_csv(path)
            self.assertEqual(len(pred),len(self.val))
            self.assertEqual(pred.example_id.tolist(),self.val.example_id.tolist())
            self.assertTrue(np.array_equal(pred.label.to_numpy(),self.val.label.to_numpy()))
            self.assertEqual(set(pred.prediction.unique())-{0,1},set(),rid)
            self.assertEqual(pred.example_id.nunique(),len(pred))

    def test_e_metrics_and_confusion_recomputed_from_saved_predictions(self):
        for rid in RUNS:
            r=self.get(rid)
            p=pd.read_csv(ROOT/r.predictions_path)
            y=p.label.to_numpy()
            yhat=p.prediction.to_numpy()
            expected={
                "val_accuracy":accuracy_score(y,yhat),
                "val_precision":precision_score(y,yhat),
                "val_recall":recall_score(y,yhat),
                "val_f1":f1_score(y,yhat),
            }
            for k,v in expected.items():
                self.assertAlmostEqual(float(r[k]),float(v),places=9,msg=f"{rid}:{k}")
            matrix=json.loads(r.confusion_matrix)
            self.assertEqual(matrix,confusion_matrix(y,yhat,labels=[0,1]).tolist())
            self.assertEqual(sum(sum(row) for row in matrix),len(y))

    def test_f_model_artifacts_exist_and_nonempty(self):
        for rid in RUNS:
            r=self.get(rid)
            artifact=ROOT/r.model_path
            self.assertTrue(artifact.is_file(),f"{rid} model missing")
            self.assertGreater(artifact.stat().st_size,1000)
            self.assertGreater(float(r.train_time_sec),0)
            self.assertGreater(float(r.inference_time_sec),0)

    def test_g_knn_svc_candidates_and_exact_classifiers(self):
        import joblib
        for rid in RUNS[:4]:
            row=self.get(rid)
            model=joblib.load(ROOT/row.model_path)
            selection=json.loads((ROOT/f"results/{rid}_candidate_selection.json").read_text())
            self.assertEqual(selection["test_rows_used"],0)
            self.assertEqual(selection["selection_only_on"],"Validation")
            self.assertEqual(len(selection["candidates"]),2)
            scores=[float(q["val_f1"]) for q in selection["candidates"]]
            self.assertAlmostEqual(max(scores),float(row.val_f1),places=9)
            hp=json.loads(row.hyperparameters)
            if rid in ("ML-01-s42","ML-03-s42"):
                from sklearn.neighbors import KNeighborsClassifier
                self.assertIsInstance(model,KNeighborsClassifier)
                self.assertEqual(model.metric,"cosine")
                self.assertEqual(model.algorithm,"brute")
                self.assertIn(int(model.n_neighbors),[3,5])
            else:
                from sklearn.svm import SVC
                self.assertIsInstance(model,SVC)
                self.assertEqual(model.kernel,"linear")
                self.assertIn(float(model.C),[0.1,1.0])

    def test_h_rnn_architecture_epochs_and_training_history(self):
        import tensorflow as tf
        for rid,recurrent in (("DL-01-s42","SimpleRNN"),
                              ("DL-02-s42","LSTM"),("DL-03-s42","GRU")):
            r=self.get(rid)
            self.assertEqual(int(r.epochs_completed),10)
            hist=pd.read_csv(ROOT/r.history_path)
            self.assertEqual(len(hist),10)
            self.assertAlmostEqual(float(hist.val_loss.iloc[-1]),float(r.val_loss),places=7)
            model=tf.keras.models.load_model(ROOT/r.model_path)
            self.assertEqual(model.count_params(),int(r.trainable_params))
            self.assertIn(recurrent,[layer.__class__.__name__ for layer in model.layers])
            layers=[layer.__class__.__name__ for layer in model.layers]
            self.assertIn("Embedding",layers)
            self.assertIn("Dense",layers)
            assert model.get_layer("embedding").mask_zero
            hp=json.loads(r.hyperparameters)
            self.assertFalse(hp["early_stopping"])
            self.assertEqual(hp["dropout"],0)
            self.assertEqual(hp["epochs_max"],10)
            self.assertEqual(hp["batch_size"],128)
            tf.keras.backend.clear_session()

    def test_i_all_figures_exist(self):
        for rid in RUNS:
            self.assertTrue((ROOT/f"results/figures/{rid}_val_confusion_matrix.png").is_file())
        for rid in RUNS[-3:]:
            self.assertTrue((ROOT/f"results/figures/{rid}_learning_curve.png").is_file())

    def test_j_splits_still_match_gate_a_and_sample_count(self):
        gate=json.loads((ROOT/"results/gate_a_acceptance.json").read_text())
        self.assertEqual(gate["status"],"PASS_EXACT_CLEANED_TEXT_CROSS_SPLIT")
        self.assertEqual(len(self.val),4956)
        for n in ("train","validation","test"):
            self.assertEqual(file_hash(ROOT/f"data/processed/{n}.parquet"),
                             self.manifest["counts"][n]["sha256"])
        self.assertEqual(len(pd.read_parquet(ROOT/"data/processed/train.parquet")),19823)

if __name__=="__main__":
    unittest.main(verbosity=2)
