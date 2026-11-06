"""Common Phase 1 evaluation, provenance, atomic run registry and figures."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss
)

from src.common import ROOT, config as load_phase0, file_hash, load_splits

RESULT_DIR=ROOT/"results"
PRED_DIR=RESULT_DIR/"predictions"
FIG_DIR=RESULT_DIR/"figures"
MODEL_DIR=ROOT/"artifacts/models"
REGISTRY=RESULT_DIR/"runs.csv"
FIELDS=[
 "run_id","experiment_group","timestamp_utc","seed","dataset_version",
 "split_hash","preprocess_version","phase0_config_hash","phase1_config_hash",
 "library_versions","representation","model_name","hyperparameters",
 "trainable_params","epochs_completed","train_time_sec","inference_time_sec",
 "hardware","val_accuracy","val_precision","val_recall","val_f1","val_loss",
 "test_accuracy","test_precision","test_recall","test_f1","test_loss",
 "confusion_matrix","history_path","predictions_path","model_path","status","notes",
]

def phase1_settings():
    cfg_path=ROOT/"config/phase1_baselines.yaml"
    c=yaml.safe_load(cfg_path.read_text(encoding="utf8"))
    p0,p0sha=load_phase0()
    manifest=json.loads((ROOT/"data/processed/split_manifest.json").read_text(encoding="utf8"))
    gate=json.loads((ROOT/"results/gate_a_acceptance.json").read_text(encoding="utf8"))
    feat=json.loads((ROOT/"results/feature_metadata.json").read_text(encoding="utf8"))
    assert gate["status"]=="PASS_EXACT_CLEANED_TEXT_CROSS_SPLIT"
    assert p0sha==manifest["config_sha256"]==c["phase0_config_sha256"]
    assert manifest["split_assignments_sha256"]==c["split_assignments_sha256"]
    assert c["dataset_variant"]==manifest["version"]=="dedup_v1.1"
    assert c["test_policy"]=="do_not_evaluate_in_phase1"
    assert feat["manifest_split_sha256"]==manifest["split_assignments_sha256"]
    for folder in (PRED_DIR,FIG_DIR,MODEL_DIR):folder.mkdir(parents=True,exist_ok=True)
    return c,file_hash(cfg_path),manifest

def subset_data():
    # Gate B deliberately does not load the official Test rows at all.
    return (pd.read_parquet(ROOT/'data/processed/train.parquet'),
            pd.read_parquet(ROOT/'data/processed/validation.parquet'))

def metrics_for_validation(y_true, predictions, val_scores=None):
    a=np.asarray(y_true,dtype=int)
    b=np.asarray(predictions,dtype=int)
    assert a.shape==b.shape
    assert set(np.unique(a))=={0,1}
    assert set(np.unique(b)).issubset({0,1})
    cm=confusion_matrix(a,b,labels=[0,1]).tolist()
    assert sum(sum(row) for row in cm)==len(a)
    return {
      "val_accuracy":float(accuracy_score(a,b)),
      "val_precision":float(precision_score(a,b,zero_division=0)),
      "val_recall":float(recall_score(a,b,zero_division=0)),
      "val_f1":float(f1_score(a,b,zero_division=0)),
      "confusion_matrix":cm,
    }

def store_pred(run_id,frame,predictions,scores=None,score_kind=""):
    out=PRED_DIR/f"{run_id}_validation.csv"
    assert len(frame)==len(predictions)
    payload=pd.DataFrame({
      "example_id":frame.example_id.to_numpy(),
      "label":frame.label.to_numpy(dtype=int),
      "prediction":np.asarray(predictions,dtype=int),
    })
    if scores is not None:
      payload[score_kind]=np.asarray(scores,dtype=float)
    payload.to_csv(out,index=False,lineterminator="\n")
    return str(out.relative_to(ROOT))

def plot_confusion(run_id, cm):
    c=np.asarray(cm,dtype=int)
    fig,ax=plt.subplots(figsize=(4.8,4.1),dpi=160)
    im=ax.imshow(c,cmap="Blues",interpolation="nearest")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Negative","Positive"],
           yticklabels=["Negative","Positive"],xlabel="Predicted",ylabel="True",
           title=f"{run_id} — Validation confusion matrix")
    for i in range(2):
      for j in range(2):
        ax.text(j,i,f"{c[i,j]:,}",ha="center",va="center",
                color="white" if c[i,j]>c.max()/2 else "black",fontsize=12)
    fig.colorbar(im,ax=ax,shrink=.8)
    fig.tight_layout()
    target=FIG_DIR/f"{run_id}_val_confusion_matrix.png"
    fig.savefig(target)
    plt.close(fig)
    return str(target.relative_to(ROOT))

def plot_history(run_id,hist):
    h=pd.DataFrame(hist)
    fig,ax=plt.subplots(figsize=(7,4.6),dpi=160)
    for key in ("loss","val_loss"):
      if key in h:ax.plot(np.arange(1,len(h)+1),h[key].to_numpy(),marker="o",label=key)
    ax.set(xlabel="Epoch",ylabel="Binary cross-entropy",title=f"{run_id} — Train vs Validation")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    out=FIG_DIR/f"{run_id}_learning_curve.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out.relative_to(ROOT))

def record_run(record):
    import fcntl  # Linux ai-server: serialize concurrent ML/CPU and DL/GPU writers.
    assert record["run_id"] and record["status"]=="validation_complete"
    assert str(record["dataset_version"])=="dedup_v1.1"
    assert all(record.get(k) in ('', None) for k in
               ('test_accuracy','test_precision','test_recall','test_f1','test_loss'))
    lockfile=REGISTRY.parent/'.runs.csv.lock'
    with lockfile.open('a+',encoding='utf-8') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        rows=[]
        if REGISTRY.exists():
            with REGISTRY.open(newline="",encoding="utf-8") as f:
                reader=csv.DictReader(f)
                assert reader.fieldnames==FIELDS
                rows=list(reader)
            if any(row["run_id"]==record["run_id"] for row in rows):
                raise RuntimeError(f"Refuse to replace an existing run ID {record['run_id']}")
        row={k:record.get(k,"") for k in FIELDS}
        for k in ("confusion_matrix","hyperparameters","library_versions"):
            if isinstance(row.get(k),(list,dict)):
                row[k]=json.dumps(row[k],ensure_ascii=False,sort_keys=True)
        rows.append(row)
        with tempfile.NamedTemporaryFile("w",delete=False,dir=REGISTRY.parent,encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            target=f.name
        os.replace(target,REGISTRY)
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
    return REGISTRY

def base_record(run_id,group,representation,model_name,hyperparams,phase1sha,manifest,hardware="CPU"):
    return dict(
        run_id=run_id,
        experiment_group=group,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        seed=42,
        dataset_version="dedup_v1.1",
        split_hash=manifest["split_assignments_sha256"],
        preprocess_version=manifest["preprocessing_version"],
        phase0_config_hash=manifest["config_sha256"],
        phase1_config_hash=phase1sha,
        library_versions={"python":platform.python_version(),
                          "numpy":np.__version__,"scikit-learn":sklearn.__version__},
        representation=representation,
        model_name=model_name,
        hyperparameters=hyperparams,
        trainable_params="",
        epochs_completed="",
        train_time_sec="",
        inference_time_sec="",
        hardware=hardware,
        val_loss="",
        test_accuracy="",
        test_precision="",
        test_recall="",
        test_f1="",
        test_loss="",
        history_path="",
        predictions_path="",
        model_path="",
        status="validation_complete",
        notes="Phase 1 uses Validation only; Test held for final Gate E.",
    )
