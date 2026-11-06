"""Phase 2 v1.1 full-data GPU training, 7 new models; never evaluate Test."""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.common import ROOT,file_hash
from src.evaluation import (phase1_settings,subset_data,
    base_record,metrics_for_validation,plot_history,plot_confusion,
    store_pred,record_run,REGISTRY,MODEL_DIR,RESULT_DIR)
from src.phase2_factory import ADVANCED,build_phase2

ALL_NEW=["DL-04","DL-05","DL-06","DL-07","DL-08","DL-09","DL-10"]

def read_config():
    config_path=ROOT/"config/phase2_conditions.yaml"
    c=yaml.safe_load(config_path.read_text(encoding="utf8"))
    base,base_sha,manifest=phase1_settings()
    assert c["version"]=="phase2-control-v1"
    assert c["dataset_variant"]=="dedup_v1.1"
    assert c["phase1_config_sha256"]==base_sha
    assert c["split_assignments_sha256"]==manifest["split_assignments_sha256"]
    assert c["test_policy"]=="do_not_evaluate_until_gate_e"
    a,b=c["shared"],base["deep_learning"]
    for key in ("embedding_dim","hidden_units","batch_size","epochs",
                "learning_rate","mask_zero","train_shuffle"):
        assert a[key]==b[key],f"Phase 2 mismatched baseline hyperparam: {key}"
    assert a["vocab_size"]==12000 and a["sequence_length"]==160
    assert a["threshold"]==b["primary_threshold"]
    assert c["regularization"]["baseline"]=="DL-02-s42"
    assert c["regularization"]["patience"]==3
    assert c["regularization"]["monitor"]=="val_loss"
    assert c["regularization"]["restore_best_weights"] is True
    assert abs(float(c["regularization"]["dropout"])-0.3)<1e-12
    reg=c["regularization"]["conditions"]
    assert reg=={
       "DL-04":{"dropout":True,"early_stopping":False},
       "DL-05":{"dropout":False,"early_stopping":True},
       "DL-06":{"dropout":True,"early_stopping":True}}
    assert c["architectures"]==ADVANCED
    original=pd.read_csv(REGISTRY)
    baseline=original[original.run_id=="DL-02-s42"]
    assert len(baseline)==1 and baseline.iloc[0].status=="validation_complete"
    return c,file_hash(config_path),base_sha,manifest

def gpu_preflight():
    devices=tf.config.list_physical_devices("GPU")
    assert devices,"No visible GPU in cnn-jupyterlab container"
    for d in devices:
        try:tf.config.experimental.set_memory_growth(d,True)
        except RuntimeError:pass
    print("GPU_PREFLIGHT_PASS",[(g.name,g.device_type) for g in devices],flush=True)

def load_trainval(smoke):
    train,val=subset_data() # deliberately never opens Test
    with np.load(ROOT/"artifacts/vectorizers/keras_token_ids_12000_160.npz",
                 allow_pickle=False) as arr:
        x=arr["X_train"]
        y=arr["y_train"]
        xv=arr["X_validation"]
        yv=arr["y_validation"]
        assert arr["ids_train"].tolist()==train.example_id.tolist()
        assert arr["ids_validation"].tolist()==val.example_id.tolist()
    assert x.shape==(19823,160) and xv.shape==(4956,160)
    assert np.array_equal(y,train.label.to_numpy())
    assert np.array_equal(yv,val.label.to_numpy())
    if smoke:
        ti=np.sort(np.concatenate([np.flatnonzero(y==i)[:256] for i in (0,1)]))
        vi=np.sort(np.concatenate([np.flatnonzero(yv==i)[:64] for i in (0,1)]))
        return train.iloc[ti],val.iloc[vi],x[ti],y[ti],xv[vi],yv[vi]
    return train,val,x,y,xv,yv

def run(run_id,smoke=False):
    cfg,phase2sha,phase1sha,manifest=read_config()
    assert run_id in ALL_NEW
    identity=f"{run_id}-s42"
    if not smoke and REGISTRY.exists():
        existing=pd.read_csv(REGISTRY)
        if identity in existing.run_id.to_numpy():
            print("ALREADY_COMPLETE_SKIPPED",identity,flush=True)
            return
    shared=cfg["shared"]
    cond=cfg["regularization"]["conditions"].get(run_id,{
        "dropout":False,"early_stopping":False})
    regularization=run_id in cfg["regularization"]["conditions"]
    if regularization:
        assert run_id not in cfg["architectures"]
    else:
        assert run_id in cfg["architectures"]
    train,val,X,y,Xv,yv=load_trainval(smoke)
    tf.keras.backend.clear_session()
    random.seed(42)
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)
    model=build_phase2(run_id,shared,cfg["regularization"]["dropout"])
    params=int(sum(np.prod(w.shape) for w in model.trainable_weights))
    model.summary(print_fn=lambda s:print(s,flush=True))
    callbacks=[]
    if cond["early_stopping"]:
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",mode="min",
            min_delta=float(cfg["regularization"]["min_delta"]),
            patience=int(cfg["regularization"]["patience"]),
            restore_best_weights=cfg["regularization"]["restore_best_weights"])]
    epochs=1 if smoke else int(shared["epochs"])
    print("PHASE2_START",identity,"smoke",smoke,
         "epoch_budget",epochs,"dropout",cond["dropout"],
         "early_stopping",cond["early_stopping"],"params",params,flush=True)
    t=time.perf_counter()
    history=model.fit(X,y,validation_data=(Xv,yv),
         epochs=epochs,batch_size=int(shared["batch_size"]),
         callbacks=callbacks,shuffle=bool(shared["train_shuffle"]),verbose=2)
    train_sec=time.perf_counter()-t
    t=time.perf_counter()
    probabilities=model.predict(Xv,batch_size=256,verbose=0).reshape(-1)
    infer_sec=time.perf_counter()-t
    pred=(probabilities>=float(shared["threshold"])).astype(np.int8)
    valid=metrics_for_validation(yv,pred)
    # Evaluate saved/restored weights, not merely last epoch's history loss.
    val_loss=float(model.evaluate(Xv,yv,batch_size=256,verbose=0)[0])
    hist={k:[float(v) for v in vals] for k,vals in history.history.items()}
    best_epoch=int(np.argmin(hist["val_loss"])+1)
    completed=len(hist["loss"])
    print("PHASE2_RESULT",identity,"epochs",completed,
          "best_val_loss_epoch",best_epoch,
          "final_post_restore_val_loss",round(val_loss,6),
          "f1",round(valid["val_f1"],6),
          "acc",round(valid["val_accuracy"],6),flush=True)
    if smoke:
        del model
        tf.keras.backend.clear_session()
        return
    history_rel=f"results/{identity}_history.csv"
    pd.DataFrame(hist).assign(epoch=np.arange(1,completed+1)).to_csv(
        ROOT/history_rel,index=False,lineterminator="\n")
    plot_history(identity,hist)
    plot_confusion(identity,valid["confusion_matrix"])
    model_relative=f"artifacts/models/{identity}.keras"
    model.save(ROOT/model_relative)
    pred_relative=store_pred(identity,val,pred,probabilities,"positive_probability")
    audit={
       "run_id":identity,"dataset_version":"dedup_v1.1",
       "split_hash":manifest["split_assignments_sha256"],
       "phase1_config_sha256":phase1sha,
       "phase2_config_sha256":phase2sha,
       "seed":42,"test_evaluations":0,
       "group":"C_dropout_early_stopping" if regularization else "D_advanced_architecture",
       "condition":cond,"architecture":run_id,
       "epoch_budget":int(shared["epochs"]),
       "epochs_completed":completed,
       "best_val_loss_epoch":best_epoch,
       "history_last_val_loss":hist["val_loss"][-1],
       "final_model_val_loss":val_loss,
       "weights_restored":bool(cond["early_stopping"] and best_epoch<completed),
       "early_stopping_callbacks":len(callbacks),
       "tensorflow_version":tf.__version__,
       "notes":"Final model metrics from final/restore_best_weights checkpoint; no official Test usage",
    }
    (ROOT/f"results/{identity}_training_metadata.json").write_text(
         json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    architecture=("LSTM + Dropout(0.3)" if run_id=="DL-04" else
                  "LSTM + EarlyStopping" if run_id=="DL-05" else
                  "LSTM + Dropout(0.3) + EarlyStopping" if run_id=="DL-06" else
                  ADVANCED[run_id])
    hparams={
       "embedding_dim":int(shared["embedding_dim"]),
       "vocab_size":int(shared["vocab_size"]),
       "sequence_length":int(shared["sequence_length"]),
       "mask_zero":bool(shared["mask_zero"]),
       "hidden_units":int(shared["hidden_units"]),
       "dropout":float(cfg["regularization"]["dropout"]) if cond["dropout"] else 0.0,
       "early_stopping":bool(cond["early_stopping"]),
       "early_stopping_patience":int(cfg["regularization"]["patience"]) if cond["early_stopping"] else None,
       "early_stopping_restore_best_weights":bool(cond["early_stopping"]),
       "optimizer":"Adam","learning_rate":float(shared["learning_rate"]),
       "batch_size":int(shared["batch_size"]),
       "epochs_max":int(shared["epochs"]),
       "loss":"binary_crossentropy","threshold":float(shared["threshold"]),
       "phase2_config_sha256":phase2sha,
       "best_val_loss_epoch":best_epoch,
       "final_post_restore_val_loss":val_loss,
    }
    record=base_record(identity,audit["group"],
         "Keras TextVectorization + trainable Embedding",architecture,
         hparams,phase1sha,manifest,
         hardware=str(tf.config.list_physical_devices("GPU")))
    record["library_versions"]["tensorflow"]=tf.__version__
    record.update(valid)
    record.update({
       "trainable_params":params,
       "epochs_completed":completed,
       "train_time_sec":round(train_sec,4),
       "inference_time_sec":round(infer_sec,4),
       "val_loss":val_loss,
       "history_path":history_rel,
       "predictions_path":pred_relative,
       "model_path":model_relative,
       "notes":("Phase 2; architecture-only fixed 10 epochs." if not regularization
          else "Phase 2 LSTM: dropout and/or early stopping as configured; final val_loss from restored weights.")
    })
    record_run(record)
    print("PHASE2_PERSISTED",identity,"val_f1",round(valid["val_f1"],6),
          "val_accuracy",round(valid["val_accuracy"],6),"train_seconds",round(train_sec,2),
          "model",model_relative,flush=True)
    del model
    tf.keras.backend.clear_session()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--runs",nargs="+",choices=ALL_NEW,default=ALL_NEW)
    args=parser.parse_args()
    gpu_preflight()
    for x in args.runs:
        run(x,args.smoke)
    print("PHASE2_RUNNER_DONE",args.runs,flush=True)

if __name__=="__main__":
    main()
