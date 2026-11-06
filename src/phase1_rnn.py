"""Phase 1 DL-01..03: strict comparable full-data baseline, no dropout/early stopping."""
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

from src.common import ROOT,file_hash
from src.evaluation import (
    phase1_settings,subset_data,metrics_for_validation,base_record,
    store_pred,plot_confusion,plot_history,record_run,RESULT_DIR,MODEL_DIR
)
from src.model_factory import BASELINES,make_single_layer_rnn


def enable_gpu():
    gpus=tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("Phase 1 DL requires available GPU. No visible GPU in Jupyter container.")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError as e:
            print("GPU growth setting already initialized:",e,flush=True)
    print("VISIBLE_GPUS",[(g.name,g.device_type) for g in gpus],flush=True)
    return gpus


def load_arrays(smoke=False):
    train,val=subset_data()
    art=ROOT/"artifacts/vectorizers/keras_token_ids_12000_160.npz"
    with np.load(art,allow_pickle=False) as arr:
        x_train=arr["X_train"]
        y_train=arr["y_train"]
        ids_train=arr["ids_train"]
        x_val=arr["X_validation"]
        y_val=arr["y_validation"]
        ids_val=arr["ids_validation"]
    assert x_train.shape==(19823,160) and x_val.shape==(4956,160)
    assert ids_train.tolist()==train.example_id.tolist()
    assert ids_val.tolist()==val.example_id.tolist()
    assert np.array_equal(y_train,train.label.to_numpy())
    assert np.array_equal(y_val,val.label.to_numpy())
    if smoke:
        train_idx=np.sort(np.concatenate([np.flatnonzero(y_train==k)[:256] for k in (0,1)]))
        val_idx=np.sort(np.concatenate([np.flatnonzero(y_val==k)[:64] for k in (0,1)]))
        train=train.iloc[train_idx]
        val=val.iloc[val_idx]
        x_train=x_train[train_idx]
        x_val=x_val[val_idx]
        y_train=y_train[train_idx]
        y_val=y_val[val_idx]
    return (train,val,x_train,y_train,x_val,y_val)


def run(run_id,smoke=False):
    cfg,phase1sha,manifest=phase1_settings()
    settings=cfg["deep_learning"]
    assert int(settings["dropout"]*100)==0 and settings["early_stopping"] is False
    assert cfg["score_selection"]=="val_f1" and cfg["test_policy"]=="do_not_evaluate_in_phase1"
    assert settings["mask_zero"] is True and int(settings["embedding_dim"])==100
    assert int(settings["hidden_units"])==64 and int(settings["epochs"])==10
    assert settings["optimizer"]=="Adam"
    train,val,x_train,y_train,x_val,y_val=load_arrays(smoke)
    seed=int(cfg["seed"])
    tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model=make_single_layer_rnn(run_id,vocab_size=12000,input_length=160,
              embedding_dim=int(settings["embedding_dim"]),
              hidden_units=int(settings["hidden_units"]),
              learning_rate=float(settings["learning_rate"]),
              mask_zero=settings["mask_zero"])
    trainable=int(sum(np.prod(v.shape) for v in model.trainable_weights))
    print("START",run_id,BASELINES[run_id],"smoke",smoke,"train",len(x_train),
          "val",len(x_val),"trainable_params",trainable,flush=True)
    model.summary(print_fn=lambda line:print(line,flush=True))
    epoch_count=1 if smoke else int(settings["epochs"])
    start=time.perf_counter()
    history=model.fit(x_train,y_train,
          validation_data=(x_val,y_val),
          batch_size=int(settings["batch_size"]),
          epochs=epoch_count,
          shuffle=bool(settings["train_shuffle"]),
          verbose=2,
          callbacks=[])
    training_s=time.perf_counter()-start
    t=time.perf_counter()
    proba=model.predict(x_val,batch_size=256,verbose=0).reshape(-1)
    inference_s=time.perf_counter()-t
    assert proba.shape==(len(y_val),)
    predictions=(proba>=float(settings["primary_threshold"])).astype(np.int8)
    vmetrics=metrics_for_validation(y_val,predictions)
    val_loss=float(history.history["val_loss"][-1])
    if smoke:
        print("SMOKE_PASS",run_id,"params",trainable,"val_loss",round(val_loss,5),
             "val_f1",round(vmetrics["val_f1"],5),flush=True)
        tf.keras.backend.clear_session()
        return
    hdata={k:[float(v) for v in values] for k,values in history.history.items()}
    h_path=RESULT_DIR/f"{run_id}-s42_history.csv"
    pd.DataFrame(hdata).assign(epoch=np.arange(1,len(hdata["loss"])+1)).to_csv(
           h_path,index=False,lineterminator="\n")
    plot_history(f"{run_id}-s42",hdata)
    plot_confusion(f"{run_id}-s42",vmetrics["confusion_matrix"])
    model_path=MODEL_DIR/f"{run_id}-s42.keras"
    model.save(model_path)
    preds_path=store_pred(f"{run_id}-s42",val,predictions,proba,"positive_probability")
    base=base_record(f"{run_id}-s42","B_single_layer_rnn",
                     "Keras TextVectorization + trainable Embedding",
                     BASELINES[run_id],
                     {"architecture":["Embedding(12000,100,mask_zero=True)",
                                      f"{BASELINES[run_id]}(64)","Dense(1,sigmoid)"],
                      "epochs_max":int(settings["epochs"]),
                      "batch_size":int(settings["batch_size"]),
                      "optimizer":"Adam","learning_rate":float(settings["learning_rate"]),
                      "loss":"binary_crossentropy","dropout":0,
                      "early_stopping":False,"threshold":float(settings["primary_threshold"])},
                     phase1sha,manifest,hardware=str(tf.config.list_physical_devices("GPU")))
    base["library_versions"]["tensorflow"]=tf.__version__
    base.update(vmetrics)
    base.update({"trainable_params":trainable,
       "epochs_completed":len(hdata["loss"]),
       "train_time_sec":round(training_s,4),
       "inference_time_sec":round(inference_s,4),
       "val_loss":val_loss,
       "history_path":str(h_path.relative_to(ROOT)),
       "predictions_path":preds_path,
       "model_path":str(model_path.relative_to(ROOT)),
       "notes":"No dropout/early stopping; training time includes epoch-level Validation, not vectorization; TensorFlow GPU determinism not guaranteed.",
    })
    record_run(base)
    print("PERSISTED",base["run_id"],"val_accuracy",round(vmetrics["val_accuracy"],6),
          "val_f1",round(vmetrics["val_f1"],6),"val_loss",round(val_loss,6),
          "params",trainable,"epochs",len(hdata["loss"]),
          "train_seconds",round(training_s,3),flush=True)
    del model
    tf.keras.backend.clear_session()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--runs",nargs="+",choices=list(BASELINES),default=list(BASELINES))
    args=parser.parse_args()
    enable_gpu()
    for name in args.runs:
        run(name,args.smoke)
    print("RNN_BASELINES_FINISHED",args.runs,flush=True)


if __name__=="__main__":
    main()
