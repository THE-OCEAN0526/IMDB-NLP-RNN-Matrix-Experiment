"""Strict preflight for ONE-TIME locked official Test. Does not run Test predictions."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.common import ROOT,file_hash
from src.evaluation import phase1_settings

R=ROOT/"results"
PATH=R/"final_test_preflight.json"

def verify():
    assert not (R/"final_test").exists(),"A final Test directory already exists; refuse evaluation"
    assert not (R/".final_test_staging").exists(),"A staging Test directory exists; refuse rerun"
    assert not PATH.exists(),"Preflight record exists; refuse repeated execution"
    lock=json.loads((R/"model_selection_lock.json").read_text(encoding="utf8"))
    pre=json.loads((R/"phase4_pretest_acceptance.json").read_text(encoding="utf8"))
    manifest=json.loads((ROOT/"data/processed/split_manifest.json").read_text(encoding="utf8"))
    feat=json.loads((R/"feature_metadata.json").read_text(encoding="utf8"))
    gate_c=json.loads((R/"gate_c_acceptance.json").read_text(encoding="utf8"))
    _,_,derived=phase1_settings()
    assert gate_c["status"]=="PASS" and pre["status"]=="PASS_PRETEST_LOCKED"
    assert pre["gate_e_final_status"]=="PENDING_OFFICIAL_TEST"
    assert lock["status"]=="LOCKED_BEFORE_FINAL_TEST"
    assert lock["official_test_not_evaluated"] is True and lock["test_metrics"] is None
    assert lock["future_final_test_scope"]==["ML-04-s42","DL-05-s42"]
    assert lock["selected_overall"]==lock["selected_traditional_ml"]=="ML-04-s42"
    assert lock["selected_deep_learning"]=="DL-05-s42"
    assert len(lock["all14_ranking"])==len(lock["source_artifacts"])==14
    assert lock["runs_registry_sha256"]==file_hash(R/"runs.csv")
    assert lock["policy_sha256"]==file_hash(ROOT/"config/phase4_selection_policy.yaml")
    assert lock["data_split_manifest_sha256"]==file_hash(ROOT/"data/processed/split_manifest.json")
    assert lock["phase4_ranked_csv_sha256"]==file_hash(R/"phase4_ranked_validation.csv")
    assert lock["phase4_summary_sha256"]==file_hash(R/"phase4_validation_summary.json")
    assert lock["phase4_rq_analysis_sha256"]==file_hash(R/"phase4_research_questions.json")
    assert lock["gate_c_acceptance_sha256"]==file_hash(R/"gate_c_acceptance.json")
    assert lock["split_assignments_sha256"]==derived["split_assignments_sha256"]
    assert manifest["counts"]["test"]["rows"]==25000
    assert file_hash(ROOT/"data/raw/test.parquet")==manifest["source_raw_sha256"]["test"]
    assert manifest["source_raw_sha256"]["test"]=="b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da"
    assert file_hash(ROOT/"data/processed/test.parquet")==manifest["counts"]["test"]["sha256"]
    assert file_hash(ROOT/"artifacts/vectorizers/tfidf_test.npz")==feat["sparse_matrices"]["tfidf_test"]["sha256"]
    assert file_hash(ROOT/"artifacts/vectorizers/keras_token_ids_12000_160.npz")==feat["keras_token_ids_sha256"]
    assert file_hash(ROOT/"artifacts/vectorizers/keras_vocabulary_12000.txt")==feat["keras_vocabulary_sha256"]
    assert file_hash(ROOT/"artifacts/vectorizers/tfidf_vectorizer.joblib")==feat["vectorizers_sha256"]["tfidf_vectorizer.joblib"]
    assert set(lock["source_artifacts"])==set(lock["all14_ranking"])
    for rid,artifact in lock["source_artifacts"].items():
        assert file_hash(ROOT/artifact["model_path"])==artifact["model_sha256"],rid
        assert file_hash(ROOT/artifact["validation_predictions_path"])==artifact["validation_predictions_sha256"],rid
    runs=pd.read_csv(R/"runs.csv")
    assert len(runs)==14 and runs.run_id.is_unique
    assert runs[["test_accuracy","test_precision","test_recall","test_f1","test_loss"]].isna().all().all()
    assert set(runs.run_id)==set(lock["all14_ranking"])
    assert not list((R/"predictions").glob("*test*"))
    # Input existence check only: don't use Test content to revise model selection.
    report={
       "stage":"LOCKED_FINAL_TEST_PREFLIGHT",
       "status":"PASS",
       "model_selection_lock_sha256":file_hash(R/"model_selection_lock.json"),
       "validation_run_registry_sha256":file_hash(R/"runs.csv"),
       "source_test_sha256":manifest["source_raw_sha256"]["test"],
       "derived_test_sha256":manifest["counts"]["test"]["sha256"],
       "split_hash":manifest["split_assignments_sha256"],
       "tensorflow_token_ids_sha256":feat["keras_token_ids_sha256"],
       "tfidf_test_matrix_sha256":feat["sparse_matrices"]["tfidf_test"]["sha256"],
       "selected_models":[{
          "run_id":rid,
          "path":lock["source_artifacts"][rid]["model_path"],
          "sha256":lock["source_artifacts"][rid]["model_sha256"],
       } for rid in lock["future_final_test_scope"]],
       "test_evaluations_performed":0,
       "approved_scope":lock["future_final_test_scope"],
       "phase0_phase1_phase2_artifacts_preserved":True,
    }
    with PATH.open("x",encoding="utf8") as f:
       f.write(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("FINAL_TEST_PREFLIGHT_PASS")
    print("OFFICIAL_TEST_ROWS",25000,"SCOPE",report["approved_scope"])
    print("LOCK_SHA",report["model_selection_lock_sha256"])
    print("NO_FINAL_TEST_RESULTS_EXIST",True)

if __name__=="__main__":
    verify()
