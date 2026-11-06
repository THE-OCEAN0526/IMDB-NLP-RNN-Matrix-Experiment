"""Immutable-in-workflow final Test selection record: create once; verify on rerun."""
from __future__ import annotations
import json
from datetime import datetime, timezone
import pandas as pd
import yaml

from src.common import ROOT,file_hash
from src.evaluation import phase1_settings

LOCK=ROOT/"results/model_selection_lock.json"

def freeze():
    policy_path=ROOT/"config/phase4_selection_policy.yaml"
    policy=yaml.safe_load(policy_path.read_text(encoding="utf8"))
    _,_,manifest=phase1_settings()
    summary=json.loads((ROOT/"results/phase4_validation_summary.json").read_text(encoding="utf8"))
    ordered=pd.read_csv(ROOT/"results/phase4_ranked_validation.csv")
    registry=pd.read_csv(ROOT/"results/runs.csv")
    assert len(registry)==len(ordered)==14
    assert set(ordered.run_id)==set(policy["eligible_runs"])
    assert policy["final_test_protocol"]["evaluate_now"] is False
    assert ordered.overall_rank.tolist()==list(range(1,15))
    assert registry[["test_accuracy","test_precision","test_recall","test_f1","test_loss"]].isna().all().all()
    ranked=registry.sort_values(["val_f1","val_accuracy","run_id"],ascending=[False,False,True]).reset_index(drop=True)
    assert ranked.run_id.tolist()==ordered.run_id.tolist()
    allbest=ranked.iloc[0].run_id
    mlbest=ranked[ranked.run_id.str.startswith("ML")].iloc[0].run_id
    dlbest=ranked[ranked.run_id.str.startswith("DL")].iloc[0].run_id
    assert summary["choices"]=={
        "overall":allbest,"best_ml":mlbest,"best_dl":dlbest,
        "locked_finalists":[mlbest,dlbest]}
    assert allbest==mlbest=="ML-04-s42" and dlbest=="DL-05-s42"
    provenance={}
    for _,run in ranked.iterrows():
        model=ROOT/run.model_path
        predictions=ROOT/run.predictions_path
        assert model.is_file() and predictions.is_file()
        provenance[run.run_id]={
            "model_path":run.model_path,
            "model_sha256":file_hash(model),
            "validation_predictions_path":run.predictions_path,
            "validation_predictions_sha256":file_hash(predictions),
            "validation_f1":float(run.val_f1),
            "validation_accuracy":float(run.val_accuracy),
            "selection_frozen_at_config_sha256":run.phase1_config_hash,
        }
    frozen={
        "version":"phase4-model-selection-lock-v1",
        "status":"LOCKED_BEFORE_FINAL_TEST",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "stage":"pre_test",
        "dataset_variant":"dedup_v1.1",
        "seed":42,
        "source_gate_c":"PASS",
        "official_test_not_evaluated":True,
        "test_rows_intended":25000,
        "policy_sha256":file_hash(policy_path),
        "runs_registry_sha256":file_hash(ROOT/"results/runs.csv"),
        "data_split_manifest_sha256":file_hash(ROOT/"data/processed/split_manifest.json"),
        "split_assignments_sha256":manifest["split_assignments_sha256"],
        "phase4_ranked_csv_sha256":file_hash(ROOT/"results/phase4_ranked_validation.csv"),
        "phase4_rq_analysis_sha256":file_hash(ROOT/"results/phase4_research_questions.json"),
        "phase4_summary_sha256":file_hash(ROOT/"results/phase4_validation_summary.json"),
        "gate_c_acceptance_sha256":file_hash(ROOT/"results/gate_c_acceptance.json"),
        "selection_algorithm":"Maximize Validation F1; tie-break higher Validation Accuracy; final tie-break run_id ascending",
        "all14_ranking":ranked.run_id.tolist(),
        "selected_overall":allbest,
        "selected_traditional_ml":mlbest,
        "selected_deep_learning":dlbest,
        "future_final_test_scope":[mlbest,dlbest],
        "threshold_policy":"preserve existing model decision rules, no Test-based retuning",
        "final_test_operation_policy":"one planned evaluation each; publish selected models regardless of Test result; no re-selection",
        "test_metrics":None,
        "source_artifacts":provenance,
        "caveats":[
          "Only exact cleaned-text cross-split duplicates removed, not semantic near-duplicates.",
          "Official Test preserves internal duplicated comments.",
          "Only one seed 42 for all 14 conditions, no significance claim.",
          "Selection lock freezes model artifacts and source metadata logically via SHA; does not enforce filesystem immutable flags.",
        ],
    }
    content=json.dumps(frozen,ensure_ascii=False,indent=2)+"\n"
    if LOCK.exists():
        old=json.loads(LOCK.read_text(encoding="utf8"))
        new=json.loads(content)
        new["created_utc"]=old.get("created_utc")
        assert old==new,"Selection lock differs; refuse any overwrite"
        print("ALREADY_LOCKED_AND_MATCHED",LOCK)
    else:
        with LOCK.open("x",encoding="utf8") as f:
            f.write(content)
        print("LOCK_CREATED",LOCK)
    print("LOCKED_WINNERS",mlbest,dlbest,"RANKED",len(ranked),"TEST_USED",False)

if __name__=="__main__":
    freeze()
