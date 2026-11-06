"""Run exact full-data Phase 1 ML 2x2, with equal Validation-only tuning grids."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from src.common import ROOT, file_hash
from src.evaluation import (
    phase1_settings,subset_data,metrics_for_validation,base_record,
    store_pred,plot_confusion,record_run,RESULT_DIR,MODEL_DIR
)

RUNS={
   "ML-01":("bow","knn"),"ML-02":("bow","svc"),
   "ML-03":("tfidf","knn"),"ML-04":("tfidf","svc"),
}

def run(run_id,smoke=False):
    cfg,phase1sha,manifest=phase1_settings()
    tr,val=subset_data()
    if smoke:
        # IMDB source order can group labels; sample both classes for a meaningful smoke test.
        train_idx=np.sort(np.concatenate([np.flatnonzero(tr.label.to_numpy()==label)[:600] for label in (0,1)]))
        val_idx=np.sort(np.concatenate([np.flatnonzero(val.label.to_numpy()==label)[:60] for label in (0,1)]))
        tr=tr.iloc[train_idx]
        val=val.iloc[val_idx]
    else:
        assert len(tr)==19823 and len(val)==4956
    name,model_type=RUNS[run_id]
    a=ROOT/"artifacts/vectorizers"
    X_train=load_npz(a/f"{name}_train.npz")
    X_val=load_npz(a/f"{name}_validation.npz")
    if smoke:
        X_train=X_train[train_idx]
        X_val=X_val[val_idx]
    assert X_train.shape==(len(tr),12000) and X_val.shape==(len(val),12000)
    y_train=tr.label.to_numpy(dtype=np.int8)
    y_val=val.label.to_numpy(dtype=np.int8)
    kcfg=cfg["ml"][model_type]
    best=None
    candidate_log=[]
    start_all=time.perf_counter()
    if model_type=="knn":
        max_k=max(int(x) for x in kcfg["k_candidates"])
        knn=KNeighborsClassifier(n_neighbors=max_k,
              weights=kcfg["weights"],algorithm=kcfg["algorithm"],
              metric=kcfg["metric"],n_jobs=int(kcfg["n_jobs"]))
        t=time.perf_counter()
        knn.fit(X_train,y_train)
        fit_time=time.perf_counter()-t
        t=time.perf_counter()
        _,indices=knn.kneighbors(X_val,n_neighbors=max_k,return_distance=True)
        neighbor_elapsed=time.perf_counter()-t
        neighbors=y_train[indices]
        for k in [int(v) for v in kcfg["k_candidates"]]:
            preds=(neighbors[:,:k].mean(axis=1)>=0.5).astype(np.int8)
            metrics=metrics_for_validation(y_val,preds)
            candidate_log.append({"parameter":"k","value":k,"val_accuracy":metrics["val_accuracy"],
                  "val_f1":metrics["val_f1"],"train_time_sec":round(fit_time,4),
                  "inference_time_sec":round(neighbor_elapsed,4)})
            if best is None or metrics["val_f1"]>best["metrics"]["val_f1"]+1e-12:
                best={"param":k,"pred":preds,"metrics":metrics}
        model=knn.set_params(n_neighbors=best["param"])
        train_seconds=fit_time
        inference_seconds=neighbor_elapsed
        t=time.perf_counter()
        score=model.predict_proba(X_val)[:,1]
        extra_infer=time.perf_counter()-t
        # predict_proba performs one more nearest-neighbor evaluation; report full final prediction cost.
        inference_seconds=extra_infer
        best["pred"]=(score>=.5).astype(np.int8)
        assert np.array_equal(best["pred"],(neighbors[:,:best["param"]].mean(axis=1)>=.5).astype(np.int8))
        score_kind="positive_probability"
    else:
        fit_seconds=[]
        pred_seconds=[]
        for C in [float(x) for x in kcfg["C_candidates"]]:
            svc=SVC(C=C,kernel=kcfg["kernel"],probability=False,
                    cache_size=float(kcfg["cache_size_mb"]),random_state=int(cfg["seed"]))
            t=time.perf_counter()
            svc.fit(X_train,y_train)
            train_duration=time.perf_counter()-t
            t=time.perf_counter()
            pred=svc.predict(X_val)
            predict_duration=time.perf_counter()-t
            metrics=metrics_for_validation(y_val,pred)
            candidate_log.append({"parameter":"C","value":C,
                  "val_accuracy":metrics["val_accuracy"],"val_f1":metrics["val_f1"],
                  "train_time_sec":round(train_duration,4),
                  "inference_time_sec":round(predict_duration,4)})
            if best is None or metrics["val_f1"]>best["metrics"]["val_f1"]+1e-12:
                best={"param":C,"pred":pred,"metrics":metrics,"model":svc,
                      "train_time":train_duration}
            fit_seconds.append(train_duration)
            pred_seconds.append(predict_duration)
            print("CANDIDATE",run_id,"C",C,"f1",round(metrics["val_f1"],5),
                  "train_s",round(train_duration,2),flush=True)
        model=best["model"]
        train_seconds=sum(fit_seconds)
        t=time.perf_counter()
        score=model.decision_function(X_val)
        best["pred"]=model.predict(X_val)
        inference_seconds=time.perf_counter()-t
        score_kind="decision_score"
    result=metrics_for_validation(y_val,best["pred"])
    chosen=best["param"]
    print("SELECTED",run_id,"parameter",chosen,"validation F1",round(result["val_f1"],6),
          "sample_size",len(tr),len(val),"smoke",smoke,flush=True)
    if smoke: return
    selected_params={"vectorizer":name,"classifier":model_type,"selected_k_or_C":chosen,
                   "selection_metric":"val_f1","candidate_values":kcfg.get("k_candidates",kcfg.get("C_candidates")),
                   "knn_metric":kcfg.get("metric"),"kernel":kcfg.get("kernel"),
                   "fixed_feature_dim":12000}
    artifact=MODEL_DIR/f"{run_id}-s42.joblib"
    joblib.dump(model,artifact)
    candidate_path=RESULT_DIR/f"{run_id}-s42_candidate_selection.json"
    candidate_path.write_text(json.dumps({
        "run_id":run_id,"dataset":"dedup_v1.1","selection_only_on":"Validation",
        "candidates":candidate_log,"chosen_parameter":chosen,
        "hyperparameter_search_total_time_sec":round(time.perf_counter()-start_all,3),
        "test_rows_used":0
    },indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    pred_path=store_pred(f"{run_id}-s42",val,best["pred"],score,score_kind)
    plot_confusion(f"{run_id}-s42",result["confusion_matrix"])
    base=base_record(f"{run_id}-s42","A_traditional",name,
         "KNeighborsClassifier" if model_type=="knn" else "SVC",
         selected_params,phase1sha,manifest)
    base.update(result)
    base.update({
       "train_time_sec":round(train_seconds,4),
       "inference_time_sec":round(inference_seconds,4),
       "model_path":str(artifact.relative_to(ROOT)),
       "predictions_path":pred_path,
       "notes":"Train time includes all C candidate fits" if model_type=="svc"
        else "KNN fit is cheap; validation cosine brute query dominates runtime.",
    })
    record_run(base)
    print("PERSISTED",base["run_id"],"val_acc",round(result["val_accuracy"],6),
          "val_f1",round(result["val_f1"],6),"train_s",round(train_seconds,2),
          "pred_s",round(inference_seconds,2),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--runs",nargs="+",choices=list(RUNS),default=list(RUNS))
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        for rid in args.runs:
            print("START",rid,"mode","SMOKE" if args.smoke else "FULL",flush=True)
            run(rid,args.smoke)
    print("ML_MATRIX_FINISHED",args.runs,flush=True)

if __name__=="__main__":
    main()
