"""Pre-Test selection acceptance; do not open official Test data for evaluation."""
from __future__ import annotations
import json
import unittest
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,confusion_matrix

from src.common import ROOT,file_hash
from src.phase4_compare import ALIASES,load_and_check,research_questions

class Phase4Pretest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy,cls.manifest,cls.data=load_and_check()
        cls.original=pd.read_csv(ROOT/"results/runs.csv")
        cls.lock=json.loads((ROOT/"results/model_selection_lock.json").read_text(encoding="utf8"))
        cls.summary=json.loads((ROOT/"results/phase4_validation_summary.json").read_text(encoding="utf8"))
        cls.rq=json.loads((ROOT/"results/phase4_research_questions.json").read_text(encoding="utf8"))

    def test_a_gate_c_pass_14_entries_and_versions(self):
        gate=json.loads((ROOT/"results/gate_c_acceptance.json").read_text())
        self.assertEqual(gate["status"],"PASS")
        self.assertEqual(gate["total_conditions"],14)
        self.assertEqual(len(self.data),14)
        self.assertEqual(set(self.data.run_id),set(ALIASES))
        self.assertEqual(self.data.run_id.nunique(),14)
        self.assertEqual(self.data.dataset_version.unique().tolist(),["dedup_v1.1"])

    def test_b_predeclared_policy_is_validation_only(self):
        self.assertEqual(self.policy["version"],"phase4-pretest-v1")
        self.assertEqual(self.policy["selection"]["primary_score"],"val_f1")
        self.assertEqual(self.policy["selection"]["descending_tie_break"],"val_accuracy")
        self.assertEqual(self.policy["selection"]["final_tie_break"],"lexicographic_run_id")
        self.assertFalse(self.policy["final_test_protocol"]["evaluate_now"])
        self.assertEqual(self.policy["final_test_protocol"]["final_evaluation_scope"],
                         "exactly_two_locked_family_winners")
        self.assertEqual(self.policy["reporting"]["optional_glove_runs"],"not_in_scope")

    def test_c_ranking_recomputed_from_original_registry(self):
        sorted_data=self.original.sort_values(
            ["val_f1","val_accuracy","run_id"],ascending=[False,False,True])
        self.assertEqual(sorted_data.run_id.tolist(),self.data.run_id.tolist())
        self.assertEqual(self.data.overall_rank.tolist(),list(range(1,15)))
        saved=pd.read_csv(ROOT/"results/phase4_ranked_validation.csv")
        self.assertEqual(saved.run_id.tolist(),sorted_data.run_id.tolist())
        self.assertTrue(np.allclose(saved.val_f1.to_numpy(),sorted_data.val_f1.to_numpy(),atol=1e-11))
        self.assertTrue(np.allclose(saved.val_accuracy.to_numpy(),sorted_data.val_accuracy.to_numpy(),atol=1e-11))

    def test_d_no_test_metrics_and_no_test_predictions(self):
        cols=["test_accuracy","test_precision","test_recall","test_f1","test_loss"]
        self.assertTrue(self.original[cols].isna().all().all())
        self.assertTrue(self.lock["official_test_not_evaluated"])
        self.assertIsNone(self.lock["test_metrics"])
        self.assertFalse((ROOT/"results/final_test").exists())
        self.assertFalse(list((ROOT/"results/predictions").glob("*test*")))
        self.assertFalse(self.summary["test_evaluations"])

    def test_e_champion_choice_and_full_ranking_are_frozen(self):
        self.assertEqual(self.lock["status"],"LOCKED_BEFORE_FINAL_TEST")
        self.assertEqual(self.lock["all14_ranking"],self.data.run_id.tolist())
        self.assertEqual(self.lock["selected_overall"],"ML-04-s42")
        self.assertEqual(self.lock["selected_traditional_ml"],"ML-04-s42")
        self.assertEqual(self.lock["selected_deep_learning"],"DL-05-s42")
        self.assertEqual(self.lock["future_final_test_scope"],["ML-04-s42","DL-05-s42"])
        self.assertEqual(self.summary["choices"]["locked_finalists"],
                         self.lock["future_final_test_scope"])

    def test_f_lock_input_files_sha256_match(self):
        paths={
            "policy_sha256":"config/phase4_selection_policy.yaml",
            "runs_registry_sha256":"results/runs.csv",
            "data_split_manifest_sha256":"data/processed/split_manifest.json",
            "phase4_ranked_csv_sha256":"results/phase4_ranked_validation.csv",
            "phase4_rq_analysis_sha256":"results/phase4_research_questions.json",
            "phase4_summary_sha256":"results/phase4_validation_summary.json",
            "gate_c_acceptance_sha256":"results/gate_c_acceptance.json",
        }
        for key,path in paths.items():
            self.assertEqual(self.lock[key],file_hash(ROOT/path),path)

    def test_g_all14_model_prediction_hashes(self):
        self.assertEqual(len(self.lock["source_artifacts"]),14)
        for rid,info in self.lock["source_artifacts"].items():
            self.assertEqual(file_hash(ROOT/info["model_path"]),info["model_sha256"],rid)
            self.assertEqual(file_hash(ROOT/info["validation_predictions_path"]),
                             info["validation_predictions_sha256"],rid)
            self.assertEqual(info["validation_f1"],
                             float(self.data.set_index("run_id").loc[rid,"val_f1"]))

    def test_h_validation_metrics_resampled_from_predictions(self):
        # Independent actual metrics computation, does not load official Test.
        val=pd.read_parquet(ROOT/"data/processed/validation.parquet")
        for _,run in self.data.iterrows():
            p=pd.read_csv(ROOT/run.predictions_path)
            self.assertEqual(len(p),4956)
            self.assertEqual(p.example_id.tolist(),val.example_id.tolist())
            self.assertEqual(p.label.tolist(),val.label.tolist())
            y=p.label;pred=p.prediction
            checks={"val_f1":f1_score(y,pred),"val_accuracy":accuracy_score(y,pred),
                "val_precision":precision_score(y,pred),"val_recall":recall_score(y,pred)}
            for key,score in checks.items():
                self.assertAlmostEqual(float(run[key]),float(score),places=10)
            self.assertEqual(json.loads(run.confusion_matrix),
                             confusion_matrix(y,pred,labels=[0,1]).tolist())

    def test_i_five_research_questions_exact(self):
        recomputed=research_questions(self.data)
        for name in ["RQ1","RQ2","RQ3","RQ4","RQ5"]:
            self.assertEqual(recomputed[name],self.rq[name],name)
        self.assertAlmostEqual(self.rq["RQ1"]["TFIDF_vs_BoW_given_KNN"],12.2488)
        self.assertAlmostEqual(self.rq["RQ4"]["ES_only_vs_baseline"],2.6723)

    def test_j_all_required_figures_and_representative_diagnostics(self):
        self.assertEqual(len(self.summary["figures"]),5)
        for name in self.summary["figures"]:
            file=ROOT/name
            self.assertTrue(file.is_file(),name)
            self.assertGreater(file.stat().st_size,5000)
        for runid in ["ML-04-s42","DL-05-s42"]:
            cm=ROOT/f"results/figures/{runid}_val_confusion_matrix.png"
            self.assertTrue(cm.is_file())
        for runid in ["DL-01-s42","DL-02-s42","DL-03-s42","DL-05-s42"]:
            self.assertTrue((ROOT/f"results/figures/{runid}_learning_curve.png").is_file())

    def test_k_notebook07_executed_no_errors_and_reports(self):
        notebook=nbformat.read(ROOT/"notebooks/07_IMDB_Results_Comparison.ipynb",as_version=4)
        code=[cell for cell in notebook.cells if cell.cell_type=="code"]
        self.assertEqual(len(code),10)
        self.assertTrue(all(c.execution_count is not None for c in code))
        errors=[item for c in code for item in c.get("outputs",[])
                if item.get("output_type")=="error"]
        self.assertFalse(errors)
        self.assertTrue((ROOT/"results/PHASE4_PRETEST_REPORT.md").is_file())
        self.assertTrue((ROOT/"results/FINAL_TEST_PROTOCOL.md").is_file())

    def test_l_existing_gate_b_models_unchanged(self):
        before=json.loads((ROOT/"results/phase2_gate_b_frozen_before.json").read_text())
        self.assertEqual(before["run_count"],7)
        for rid,info in before["models_and_registry_rows"].items():
            self.assertEqual(file_hash(ROOT/info["model_path"]),info["model_sha256"])
        self.assertEqual(file_hash(ROOT/"results/gate_b_acceptance.json"),
                         before["gate_b_acceptance_sha256"])

if __name__=="__main__":
    unittest.main(verbosity=2)
