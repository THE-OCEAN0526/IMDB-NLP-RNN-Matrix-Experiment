"""One-time official IMDB Test for pre-locked ML-04 and DL-05 ONLY.

Run after strict preflight. Crash leaves .final_test_staging, blocking silent
repeat. On success atomically rename staging -> final_test. Never modify runs.csv.
"""
from __future__ import annotations
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import scipy
from scipy.sparse import load_npz
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss
)
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common import ROOT, file_hash

RESULTS=ROOT/"results"
STAGE=RESULTS/".final_test_staging"
FINAL=RESULTS/"final_test"
PREFLIGHT=RESULTS/"final_test_preflight.json"

def metrics(y,pred):
    cm=confusion_matrix(y,pred,labels=[0,1])
    return {
       "accuracy":float(accuracy_score(y,pred)),
       "precision":float(precision_score(y,pred,zero_division=0)),
       "recall":float(recall_score(y,pred,zero_division=0)),
       "f1":float(f1_score(y,pred,zero_division=0)),
       "confusion_matrix":cm.tolist(),
    }

def plot_confusion(model_name,rid,cm,save):
    m=np.asarray(cm,dtype=int)
    fig,ax=plt.subplots(figsize=(5,4.4),dpi=160)
    image=ax.imshow(m,cmap="Blues",interpolation="nearest")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Negative","Positive"],
           yticklabels=["Negative","Positive"],xlabel="Predicted label",
           ylabel="True label",title=model_name+" — official Test")
    for i in range(2):
       for j in range(2):
           ax.text(j,i,f"{m[i,j]:,}",ha="center",va="center",
              color="white" if m[i,j]>m.max()/2 else "black",fontsize=12)
    fig.colorbar(image,ax=ax,shrink=.8)
    fig.tight_layout();fig.savefig(save);plt.close(fig)

def main():
    # Verify all approval/lock artifacts before first Test inference.
    assert not FINAL.exists(),"Final Test already exists, refusing another run"
    assert not STAGE.exists(),"Incomplete staging exists, refusing another run"
    assert PREFLIGHT.is_file(),"Must run read-only model selection preflight first"
    pf=json.loads(PREFLIGHT.read_text(encoding="utf8"))
    lock=json.loads((RESULTS/"model_selection_lock.json").read_text(encoding="utf8"))
    manifest=json.loads((ROOT/"data/processed/split_manifest.json").read_text(encoding="utf8"))
    feat=json.loads((RESULTS/"feature_metadata.json").read_text(encoding="utf8"))
    assert pf["status"]=="PASS" and pf["test_evaluations_performed"]==0
    assert lock["status"]=="LOCKED_BEFORE_FINAL_TEST"
    assert lock["official_test_not_evaluated"] is True
    assert lock["future_final_test_scope"]==pf["approved_scope"]==["ML-04-s42","DL-05-s42"]
    assert pf["model_selection_lock_sha256"]==file_hash(RESULTS/"model_selection_lock.json")
    assert pf["validation_run_registry_sha256"]==file_hash(RESULTS/"runs.csv")
    assert pf["source_test_sha256"]==manifest["source_raw_sha256"]["test"]
    assert pf["derived_test_sha256"]==manifest["counts"]["test"]["sha256"]
    assert file_hash(ROOT/"data/raw/test.parquet")==pf["source_test_sha256"]
    assert file_hash(ROOT/"data/processed/test.parquet")==pf["derived_test_sha256"]
    assert file_hash(ROOT/"artifacts/vectorizers/tfidf_test.npz")==feat["sparse_matrices"]["tfidf_test"]["sha256"]
    assert file_hash(ROOT/"artifacts/vectorizers/keras_token_ids_12000_160.npz")==feat["keras_token_ids_sha256"]
    registry=pd.read_csv(RESULTS/"runs.csv")
    assert len(registry)==14 and registry.run_id.is_unique
    assert registry[["test_accuracy","test_precision","test_recall","test_f1","test_loss"]].isna().all().all()
    for rid in pf["approved_scope"]:
        item=lock["source_artifacts"][rid]
        assert file_hash(ROOT/item["model_path"])==item["model_sha256"]
    # One staging directory: if any failure after Test use, leave it for review; never retry blindly.
    STAGE.mkdir(exist_ok=False)
    (STAGE/"EVALUATION_IN_PROGRESS.json").write_text(json.dumps({
        "stage":"OFFICIAL_TEST_IN_PROGRESS",
        "started_at_utc":datetime.now(timezone.utc).isoformat(),
        "locked_selection_sha256":pf["model_selection_lock_sha256"],
        "original_test_source_sha256":pf["source_test_sha256"],
        "scope":pf["approved_scope"],
        "policy":"A partial failure must be reviewed; do not rerun or change model after seeing Test."
    },indent=2)+"\n",encoding="utf8")
    print("OFFICIAL_TEST_ONCE_STARTED",datetime.now(timezone.utc).isoformat(),flush=True)
    test=pd.read_parquet(ROOT/"data/processed/test.parquet")
    assert len(test)==25000
    assert test.source_split.eq("test").all()
    assert test.source_index.tolist()==list(range(25000))
    assert test.example_id.is_unique
    assert test.label.value_counts().sort_index().to_dict()=={0:12500,1:12500}
    y=test.label.to_numpy(dtype=np.int8)
    records={}
    pred_hashes={}
    # ML-04: original SVC decision policy (no change to C, feature weights or threshold).
    rid="ML-04-s42"
    model_path=ROOT/lock["source_artifacts"][rid]["model_path"]
    svc=joblib.load(model_path)
    assert isinstance(svc,SVC) and svc.kernel=="linear"
    assert abs(float(svc.C)-1.0)<1e-12
    matrix=load_npz(ROOT/"artifacts/vectorizers/tfidf_test.npz")
    assert matrix.shape==(25000,12000)
    t=time.perf_counter()
    ml_pred=np.asarray(svc.predict(matrix),dtype=np.int8)
    ml_scores=np.asarray(svc.decision_function(matrix),dtype=float)
    ml_inference=time.perf_counter()-t
    assert ml_pred.shape==(25000,) and ml_scores.shape==(25000,)
    assert (ml_pred==(ml_scores>=0).astype(np.int8)).all()
    ml_stat=metrics(y,ml_pred)
    ml_stat.update({
       "run_id":rid,"model":"TF-IDF + SVC","test_loss":None,
       "inference_time_sec":round(ml_inference,4),
       "prediction_policy":"Original sklearn.svm.SVC.predict, no new threshold",
       "test_size":25000,
       "model_sha256":file_hash(model_path),
       "feature_matrix_sha256":file_hash(ROOT/"artifacts/vectorizers/tfidf_test.npz"),
    })
    ml_csv=STAGE/f"{rid}_test_predictions.csv"
    pd.DataFrame({
       "example_id":test.example_id.to_numpy(),
       "source_index":test.source_index.to_numpy(),
       "label":y,
       "prediction":ml_pred,
       "decision_score":ml_scores,
    }).to_csv(ml_csv,index=False,lineterminator="\n",float_format="%.12g")
    plot_confusion("TF-IDF + SVC",rid,ml_stat["confusion_matrix"],
                   STAGE/f"{rid}_test_confusion_matrix.png")
    ml_stat["predictions_csv"]=ml_csv.name
    ml_stat["predictions_sha256"]=file_hash(ml_csv)
    records[rid]=ml_stat
    print("OFFICIAL_TEST_ML_COMPLETE",rid,"accuracy",round(ml_stat["accuracy"],6),
          "f1",round(ml_stat["f1"],6),"seconds",round(ml_inference,2),flush=True)

    # DL-05: from pre-frozen early-stopped Keras model, fixed p>=0.5.
    rid="DL-05-s42"
    token_path=ROOT/"artifacts/vectorizers/keras_token_ids_12000_160.npz"
    with np.load(token_path,allow_pickle=False) as data:
        X=np.array(data["X_test"])
        y_tokens=np.array(data["y_test"])
        ids=np.array(data["ids_test"])
    assert X.shape==(25000,160)
    assert ids.tolist()==test.example_id.tolist()
    assert np.array_equal(y_tokens,y)
    assert np.min(X)>=0 and np.max(X)<12000
    model_path=ROOT/lock["source_artifacts"][rid]["model_path"]
    gpus=tf.config.list_physical_devices("GPU")
    assert gpus, "DL-05 final evaluation should use ai-server GPU"
    for gpu in gpus:
        try:tf.config.experimental.set_memory_growth(gpu,True)
        except RuntimeError:pass
    model=tf.keras.models.load_model(model_path)
    assert model.count_params()==1242305
    assert isinstance(model.get_layer("lstm"),tf.keras.layers.LSTM)
    assert model.get_layer("embedding").mask_zero
    assert not any(isinstance(layer,tf.keras.layers.Dropout) for layer in model.layers)
    t=time.perf_counter()
    positive_probability=model.predict(X,batch_size=256,verbose=0).reshape(-1)
    dl_inference=time.perf_counter()-t
    assert positive_probability.shape==(25000,)
    assert np.isfinite(positive_probability).all()
    dl_pred=(positive_probability>=0.5).astype(np.int8)
    dl_stat=metrics(y,dl_pred)
    # This is binary cross-entropy on the already-produced probabilities; not a second fit.
    bce=float(log_loss(y,positive_probability,labels=[0,1]))
    dl_stat.update({
       "run_id":rid,"model":"LSTM + EarlyStopping","test_loss":bce,
       "test_loss_definition":"sklearn.log_loss on fixed Keras sigmoid output (binary cross-entropy)",
       "inference_time_sec":round(dl_inference,4),
       "prediction_policy":"Fixed model sigmoid score >= 0.5",
       "test_size":25000,
       "model_sha256":file_hash(model_path),
       "token_ids_sha256":file_hash(token_path),
       "trainable_params":int(model.count_params()),
    })
    dl_csv=STAGE/f"{rid}_test_predictions.csv"
    pd.DataFrame({
       "example_id":test.example_id.to_numpy(),
       "source_index":test.source_index.to_numpy(),
       "label":y,
       "prediction":dl_pred,
       "positive_probability":positive_probability,
    }).to_csv(dl_csv,index=False,lineterminator="\n",float_format="%.12g")
    plot_confusion("LSTM + EarlyStopping",rid,dl_stat["confusion_matrix"],
                   STAGE/f"{rid}_test_confusion_matrix.png")
    dl_stat["predictions_csv"]=dl_csv.name
    dl_stat["predictions_sha256"]=file_hash(dl_csv)
    records[rid]=dl_stat
    print("OFFICIAL_TEST_DL_COMPLETE",rid,"accuracy",round(dl_stat["accuracy"],6),
          "f1",round(dl_stat["f1"],6),"bce",round(bce,6),
          "seconds",round(dl_inference,2),flush=True)
    del model
    tf.keras.backend.clear_session()

    # Comparison plot is presentation only; selection remains Validation-locked.
    subset=registry.set_index("run_id").loc[pf["approved_scope"]]
    chart_name="finalists_validation_vs_test.png"
    fig,axes=plt.subplots(figsize=(7.5,4.2),dpi=165)
    x=np.arange(2)
    axes.bar(x-.18,subset.val_f1.to_numpy(),width=.35,label="Validation F1")
    axes.bar(x+.18,[records[r]["f1"] for r in pf["approved_scope"]],width=.35,label="Official Test F1")
    axes.set_xticks(x,["ML-04 / TF-IDF + SVC","DL-05 / LSTM + ES"])
    axes.set(ylim=(0,1),ylabel="Binary F1",title="Preselected finalists — Validation vs one-time Test")
    axes.grid(axis="y",alpha=.2);axes.legend()
    fig.tight_layout();fig.savefig(STAGE/chart_name);plt.close(fig)

    body={
      "schema_version":"gate-e-final-test-v1",
      "status":"OFFICIAL_TEST_EVALUATED_ONCE",
      "completed_at_utc":datetime.now(timezone.utc).isoformat(),
      "dataset_variant":"dedup_v1.1",
      "official_test_rows":25000,
      "test_source_sha256":pf["source_test_sha256"],
      "test_processed_sha256":pf["derived_test_sha256"],
      "split_hash":pf["split_hash"],
      "model_selection_lock_sha256":pf["model_selection_lock_sha256"],
      "baseline_runs_registry_sha256":pf["validation_run_registry_sha256"],
      "preflight_sha256":file_hash(PREFLIGHT),
      "selection_was_validation_only":True,
      "scope":["ML-04-s42","DL-05-s42"],
      "model_evaluations_performed":2,
      "test_based_retuning":False,
      "predictions_and_metrics":records,
      "comparison_figure":chart_name,
      "test_note":"Official Test internal duplicate reviews remain; no post-Test model changes.",
      "environment":{"python":platform.python_version(),
                     "tensorflow":tf.__version__,"scipy":scipy.__version__,
                     "numpy":np.__version__,"pandas":pd.__version__,
                     "gpu_devices":[g.name for g in gpus]},
    }
    metric_name="locked_final_test_metrics.json"
    (STAGE/metric_name).write_text(json.dumps(body,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    file_manifest={
       "stage":"official_test",
       "model_selection_lock_sha256":pf["model_selection_lock_sha256"],
       "metrics_file":metric_name,
       "metrics_sha256":file_hash(STAGE/metric_name),
       "output_files_sha256":{
           file.name:file_hash(file) for file in sorted(STAGE.iterdir()) if file.is_file()
       },
    }
    (STAGE/"final_test_output_manifest.json").write_text(
        json.dumps(file_manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    assert not FINAL.exists()
    STAGE.rename(FINAL)
    print("ONE_TIME_OFFICIAL_TEST_OUTPUT_COMMITTED",FINAL,flush=True)
    print("FINAL_RESULTS",json.dumps({
       rid:{k:round(value,7) for k,value in m.items() if k in ("accuracy","precision","recall","f1")}
       for rid,m in records.items()},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
