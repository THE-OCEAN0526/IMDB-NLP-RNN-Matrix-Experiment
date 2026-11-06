"""Gate C: 14 validated conditions, 7 prior models unchanged, Test still sealed."""
from __future__ import annotations
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

from src.common import ROOT,file_hash
from src.evaluation import phase1_settings
from src.phase2_factory import ADVANCED

PRIOR=[f"ML-{i:02d}-s42" for i in range(1,5)]+[f"DL-{i:02d}-s42" for i in range(1,4)]
NEW=[f"DL-{i:02d}-s42" for i in range(4,11)]
ALL=PRIOR+NEW

class GateC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base,cls.base_hash,cls.manifest=phase1_settings()
        cls.p2=yaml.safe_load((ROOT/"config/phase2_conditions.yaml").read_text())
        cls.p2_hash=file_hash(ROOT/"config/phase2_conditions.yaml")
        cls.registry=pd.read_csv(ROOT/"results/runs.csv")
        cls.rows=cls.registry.set_index("run_id")
        cls.validation=pd.read_parquet(ROOT/"data/processed/validation.parquet")
        cls.freeze=json.loads((ROOT/"results/phase2_gate_b_frozen_before.json").read_text())

    def runrow(self,rid):
        return self.rows.loc[rid]

    def test_a_14_unique_baselines_and_new_conditions(self):
        self.assertEqual(set(self.registry.run_id),set(ALL))
        self.assertEqual(self.registry.run_id.nunique(),14)
        self.assertTrue((self.registry.status=="validation_complete").all())

    def test_b_frozen_config_dataset_versions(self):
        self.assertEqual(self.p2["version"],"phase2-control-v1")
        self.assertEqual(self.p2["phase1_config_sha256"],self.base_hash)
        self.assertEqual(self.p2["split_assignments_sha256"],self.manifest["split_assignments_sha256"])
        self.assertEqual(self.p2["test_policy"],"do_not_evaluate_until_gate_e")
        for _,r in self.registry.iterrows():
            self.assertEqual(r.dataset_version,"dedup_v1.1")
            self.assertEqual(r.split_hash,self.manifest["split_assignments_sha256"])
            self.assertEqual(r.phase0_config_hash,self.manifest["config_sha256"])
            self.assertEqual(r.phase1_config_hash,self.base_hash)
            self.assertEqual(int(r.seed),42)

    def test_c_gate_b_model_hashes_unchanged(self):
        self.assertEqual(self.freeze["run_count"],7)
        self.assertEqual(set(self.freeze["models_and_registry_rows"]),set(PRIOR))
        for rid,data in self.freeze["models_and_registry_rows"].items():
            record=self.runrow(rid)
            model_path=ROOT/data["model_path"]
            self.assertEqual(file_hash(model_path),data["model_sha256"],rid)
            self.assertEqual(record.model_path,data["model_path"])
        self.assertEqual(file_hash(ROOT/"results/gate_b_acceptance.json"),
                         self.freeze["gate_b_acceptance_sha256"])

    def test_d_gate_b_registry_rows_unchanged(self):
        # pandas type inference applied consistently to prior snapshot and entire registry.
        original=pd.read_csv(ROOT/"results/runs.csv",keep_default_na=False).set_index("run_id")
        for rid,data in self.freeze["models_and_registry_rows"].items():
            record=original.loc[rid].to_dict()
            record["run_id"]=rid
            actual=hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False).encode("utf8")).hexdigest()
            self.assertEqual(actual,data["row_sha256"],rid)

    def test_e_validation_predictions_and_metrics(self):
        for rid in ALL:
            r=self.runrow(rid)
            csv=ROOT/r.predictions_path
            self.assertTrue(csv.is_file(),rid)
            p=pd.read_csv(csv)
            self.assertEqual(len(p),4956)
            self.assertEqual(p.example_id.tolist(),self.validation.example_id.tolist())
            self.assertEqual(p.label.tolist(),self.validation.label.tolist())
            y=p.label.to_numpy(); yhat=p.prediction.to_numpy()
            metric={
                "val_accuracy":accuracy_score(y,yhat),
                "val_precision":precision_score(y,yhat),
                "val_recall":recall_score(y,yhat),
                "val_f1":f1_score(y,yhat),
            }
            for key,value in metric.items():
                self.assertAlmostEqual(float(r[key]),float(value),places=10,msg=f"{rid} {key}")
            self.assertEqual(json.loads(r.confusion_matrix),
                             confusion_matrix(y,yhat,labels=[0,1]).tolist())

    def test_f_all_test_columns_blank(self):
        for col in ("test_accuracy","test_precision","test_recall","test_f1","test_loss"):
            self.assertTrue(self.registry[col].isna().all(),col)

    def test_g_new_phase2_metadata_and_artifacts(self):
        for rid in NEW:
            r=self.runrow(rid)
            metadata=json.loads((ROOT/f"results/{rid}_training_metadata.json").read_text())
            self.assertEqual(metadata["phase2_config_sha256"],self.p2_hash)
            self.assertEqual(metadata["test_evaluations"],0)
            self.assertEqual(metadata["run_id"],rid)
            self.assertEqual(metadata["epochs_completed"],int(r.epochs_completed))
            self.assertEqual(metadata["split_hash"],self.manifest["split_assignments_sha256"])
            self.assertAlmostEqual(metadata["final_model_val_loss"],float(r.val_loss),places=7)
            self.assertTrue((ROOT/r.model_path).is_file())
            self.assertTrue((ROOT/r.history_path).is_file())
            self.assertTrue((ROOT/f"results/figures/{rid}_learning_curve.png").is_file())
            self.assertTrue((ROOT/f"results/figures/{rid}_val_confusion_matrix.png").is_file())
            self.assertGreater(float(r.train_time_sec),0)
            self.assertGreater(float(r.inference_time_sec),0)

    def test_h_regularization_2x2_reuse_baseline(self):
        group=["DL-02-s42","DL-04-s42","DL-05-s42","DL-06-s42"]
        self.assertEqual(len(set(group)),4)
        regular=self.p2["regularization"]
        self.assertEqual(regular["baseline"],"DL-02-s42")
        self.assertEqual(regular["monitor"],"val_loss")
        self.assertEqual(regular["patience"],3)
        self.assertTrue(regular["restore_best_weights"])
        for rid in group[1:]:
            r=self.runrow(rid)
            meta=json.loads((ROOT/f"results/{rid}_training_metadata.json").read_text())
            cond=regular["conditions"][rid[:-4]]
            self.assertEqual(meta["condition"],cond)
            history=pd.read_csv(ROOT/r.history_path)
            self.assertEqual(len(history),int(r.epochs_completed))
            self.assertLessEqual(int(r.epochs_completed),10)
            self.assertEqual(meta["best_val_loss_epoch"],int(history.val_loss.argmin()+1))
            h=json.loads(r.hyperparameters)
            self.assertEqual(h["early_stopping"],cond["early_stopping"])
            self.assertEqual(bool(h["dropout"]>0),cond["dropout"])
            self.assertAlmostEqual(h["dropout"],0.3 if cond["dropout"] else 0.0)
            if cond["early_stopping"]:
                self.assertEqual(meta["early_stopping_callbacks"],1)
                self.assertTrue(meta["weights_restored"] if meta["best_val_loss_epoch"]<len(history) else True)
                self.assertAlmostEqual(float(r.val_loss),history.val_loss.min(),places=5)
            else:
                self.assertEqual(meta["early_stopping_callbacks"],0)
                self.assertAlmostEqual(float(r.val_loss),history.val_loss.iloc[-1],places=5)

    def test_i_advanced_architectures_and_no_regularization(self):
        for rid,name in ADVANCED.items():
            r=self.runrow(rid+"-s42")
            h=json.loads(r.hyperparameters)
            self.assertEqual(h["dropout"],0)
            self.assertFalse(h["early_stopping"])
            self.assertEqual(int(r.epochs_completed),10)
            m=tf.keras.models.load_model(ROOT/r.model_path)
            self.assertEqual(int(r.trainable_params),m.count_params())
            layer_types=[x.__class__.__name__ for x in m.layers]
            self.assertNotIn("Dropout",layer_types)
            self.assertIn("Embedding",layer_types)
            if rid=="DL-07":
                self.assertTrue(m.get_layer("lstm_1").return_sequences)
                self.assertEqual([type(m.get_layer("lstm_1")).__name__,
                                  type(m.get_layer("lstm_2")).__name__],["LSTM","LSTM"])
            elif rid=="DL-08":
                self.assertTrue(m.get_layer("lstm_1").return_sequences)
                self.assertEqual(type(m.get_layer("gru_2")).__name__,"GRU")
            else:
                b=m.get_layer("bidirectional_lstm" if rid=="DL-09" else "bidirectional_gru")
                self.assertIsInstance(b,tf.keras.layers.Bidirectional)
                self.assertEqual(type(b.forward_layer).__name__,"LSTM" if rid=="DL-09" else "GRU")
            del m
            tf.keras.backend.clear_session()

    def test_j_all_new_rnns_share_core_control_parameters(self):
        shared=self.p2["shared"]
        self.assertEqual(shared["embedding_dim"],100)
        self.assertEqual(shared["hidden_units"],64)
        self.assertEqual(shared["batch_size"],128)
        self.assertEqual(shared["epochs"],10)
        self.assertAlmostEqual(shared["learning_rate"],0.001)
        for rid in NEW:
            h=json.loads(self.runrow(rid).hyperparameters)
            for key in ("embedding_dim","vocab_size","sequence_length","mask_zero",
                        "hidden_units","batch_size","epochs_max","learning_rate"):
                self.assertEqual(h[key],shared["epochs"] if key=="epochs_max" else shared[key],(rid,key))

    def test_k_phase2_and_baseline_models_reload(self):
        for rid in ["DL-02-s42"]+NEW:
            r=self.runrow(rid)
            m=tf.keras.models.load_model(ROOT/r.model_path)
            self.assertEqual(int(r.trainable_params),m.count_params())
            self.assertIsInstance(m.get_layer("embedding"),tf.keras.layers.Embedding)
            self.assertTrue(m.get_layer("embedding").mask_zero)
            del m
            tf.keras.backend.clear_session()

    def test_l_no_changes_to_gate_a_original_data(self):
        assert self.manifest["version"]=="dedup_v1.1"
        for name in ("train","validation","test"):
            p=ROOT/f"data/processed/{name}.parquet"
            self.assertEqual(file_hash(p),self.manifest["counts"][name]["sha256"])
        self.assertEqual(len(self.validation),4956)
        self.assertEqual(len(pd.read_parquet(ROOT/"data/processed/train.parquet")),19823)

if __name__=="__main__":
    unittest.main(verbosity=2)
