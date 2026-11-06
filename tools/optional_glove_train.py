"""Two GloVe GRU conditions with identical data/architecture/budget as DL-03.

External optional extension after Gate E, Validation-only. Never opens official
Test, never updates the prior runs.csv or their existing model artifacts.
"""
from __future__ import annotations
import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

from src.common import ROOT,file_hash

OPT=ROOT/"optional_glove"
ART=OPT/"artifacts"
R=OPT/"results"
CFG=OPT/"experiment.yaml"
RUNS=["TL-01-s42","TL-02-s42"]

def preflight():
    cfg=yaml.safe_load(CFG.read_text(encoding="utf8"))
    assert cfg["version"]=="optional-glove-v1.0"
    assert cfg["test_policy"]=="Test_already_seen_in_prior_work_do_not_reevaluate_or_tune"
    assert cfg["comparison"]["official_Test_evaluation"]=="prohibited"
    assert cfg["baseline_run"]=="DL-03-s42"
    assert cfg["new_runs"]==RUNS
    assert cfg["conditions"][RUNS[0]]["embedding_trainable"] is False
    assert cfg["conditions"][RUNS[1]]["embedding_trainable"] is True
    c=cfg["common_training"]
    assert c["seed"]==42 and c["max_tokens"]==12000
    assert c["output_sequence_length"]==160 and c["embedding_dim"]==100 and c["hidden_units"]==64
    assert c["batch_size"]==128 and c["max_epochs"]==10
    assert c["learning_rate"]==0.001 and c["optimizer"]=="Adam"
    assert c["loss"]=="binary_crossentropy" and c["sigmoid_threshold"]==0.5
    assert c["dropout"]==0 and c["early_stopping"] is False
    assert c["mask_zero"] is True
    gate=json.loads((ROOT/"results/gate_e_acceptance.json").read_text())
    assert gate["status"]=="PASS"
    snapshot=json.loads((R/"FROZEN_OFFICIAL_GATE_E_BEFORE_OPTIONAL.json").read_text())
    assert snapshot["original_gate_e"]=="PASS"
    for relative,sha in snapshot["source_artifacts_sha256"].items():
        assert file_hash(ROOT/relative)==sha,relative
    provenance=json.loads((R/"glove_source_manifest.json").read_text())
    assert provenance["experiment_config_sha256"]==file_hash(CFG)
    mat_file=ROOT/provenance["aligned_matrix_path"]
    mask_file=ROOT/provenance["aligned_mask_path"]
    assert file_hash(mat_file)==provenance["aligned_matrix_sha256"]
    assert file_hash(mask_file)==provenance["aligned_mask_sha256"]
    assert file_hash(ROOT/provenance["vocabulary_path"])==provenance["vocabulary_sha256"]
    baseline=pd.read_csv(ROOT/"results/runs.csv")
    base=baseline.loc[baseline.run_id=="DL-03-s42"]
    assert len(base)==1 and abs(base.iloc[0].val_f1-0.810234)<0.0001
    return cfg,provenance

def data():
    # Gate A frozen NPZ already contains Train/Validation IDs and labels.
    # Avoid depending on pyarrow in a fresh ephemeral GPU container.
    with np.load(ROOT/"artifacts/vectorizers/keras_token_ids_12000_160.npz",
                 allow_pickle=False) as z:
        x=z["X_train"];y=z["y_train"]
        xv=z["X_validation"];yv=z["y_validation"]
        id_tr=z["ids_train"];id_val=z["ids_validation"]
    assert x.shape==(19823,160) and xv.shape==(4956,160)
    assert len(id_tr)==19823 and len(id_val)==4956
    assert len(set(id_tr.tolist()))==19823 and len(set(id_val.tolist()))==4956
    assert set(np.unique(y))=={0,1} and set(np.unique(yv))=={0,1}
    assert int(y.sum())==9954 and int(yv.sum())==2489
    val=pd.DataFrame({"example_id":id_val,"label":yv})
    return x,y,xv,yv,val

def build(cfg,matrix,trainable):
    c=cfg["common_training"]
    inp=tf.keras.Input(shape=(c["output_sequence_length"],),dtype="int32",name="token_ids")
    emb=tf.keras.layers.Embedding(
        input_dim=c["max_tokens"],
        output_dim=c["embedding_dim"],
        mask_zero=c["mask_zero"],
        trainable=trainable,name="embedding")
    x=emb(inp)
    x=tf.keras.layers.GRU(c["hidden_units"],name="gru")(x)
    p=tf.keras.layers.Dense(1,activation="sigmoid",name="positive_probability")(x)
    model=tf.keras.Model(inputs=inp,outputs=p,name="IMDB_GRU_GloVe_100d")
    # Exact pretrained weight initialization before fitting. PAD zero, OOV deterministic.
    emb.set_weights([matrix])
    assert np.array_equal(emb.get_weights()[0],matrix)
    assert emb.trainable==trainable
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=c["learning_rate"]),
      loss="binary_crossentropy",
      metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])
    return model

def score(y,p):
    r=(np.asarray(p)>=.5).astype(np.int8)
    return r,{
      "val_accuracy":float(accuracy_score(y,r)),
      "val_precision":float(precision_score(y,r,zero_division=0)),
      "val_recall":float(recall_score(y,r,zero_division=0)),
      "val_f1":float(f1_score(y,r,zero_division=0)),
      "val_confusion_matrix":confusion_matrix(y,r,labels=[0,1]).tolist(),
    }

def figures(run,hist,metrics):
    fig,ax=plt.subplots(figsize=(6.3,4.3),dpi=155)
    ax.plot(range(1,len(hist["loss"])+1),hist["loss"],label="Train BCE",marker="o")
    ax.plot(range(1,len(hist["val_loss"])+1),hist["val_loss"],label="Validation BCE",marker="o")
    ax.set(xlabel="Epoch",ylabel="Binary crossentropy",title=run+" — GloVe GRU")
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(R/f"{run}_loss.png");plt.close(fig)
    m=np.asarray(metrics["val_confusion_matrix"],dtype=int)
    fig,ax=plt.subplots(figsize=(4.6,4),dpi=155)
    image=ax.imshow(m,cmap="Blues",interpolation="nearest")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Negative","Positive"],
           yticklabels=["Negative","Positive"],xlabel="Predicted",ylabel="Actual",
           title=run+" — Validation")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(m[i,j]),ha="center",va="center",
               color="white" if m[i,j]>m.max()/2 else "black",fontsize=12)
    fig.colorbar(image,ax=ax,shrink=.8);fig.tight_layout()
    fig.savefig(R/f"{run}_val_confusion_matrix.png");plt.close(fig)

def run_one(rid,smoke=False):
    cfg,provenance=preflight()
    assert rid in RUNS
    target=R/f"{rid}.json"
    model_path=ART/f"{rid}.keras"
    pred_csv=R/f"{rid}_validation_predictions.csv"
    if not smoke:
        assert not any(p.exists() for p in
           (target,model_path,pred_csv,R/f"{rid}_history.csv",
            R/f"{rid}_loss.png",R/f"{rid}_val_confusion_matrix.png")),"Refuse overwrite existing run artifacts"
    dev=tf.config.list_physical_devices("GPU")
    assert dev,"Optional GloVe training requires working ai-server GPU (fresh isolated container)"
    for g in dev:
        try:tf.config.experimental.set_memory_growth(g,True)
        except RuntimeError:pass
    trainable=bool(cfg["conditions"][rid]["embedding_trainable"])
    matrix=np.load(ROOT/provenance["aligned_matrix_path"],allow_pickle=False)
    assert matrix.shape==(12000,100)
    assert np.all(matrix[0]==0)
    mask=np.load(ROOT/provenance["aligned_mask_path"],allow_pickle=False)
    assert len(mask)==12000 and not mask[0] and not mask[1]
    x,y,xv,yv,val=data()
    if smoke:
        it=np.sort(np.concatenate([np.flatnonzero(y==i)[:256] for i in [0,1]]))
        iv=np.sort(np.concatenate([np.flatnonzero(yv==i)[:64] for i in [0,1]]))
        x=x[it]; y=y[it]; xv=xv[iv]; yv=yv[iv]; val=val.iloc[iv]
    random.seed(42)
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)
    tf.keras.backend.clear_session()
    m=build(cfg,matrix,trainable)
    params=int(sum(np.prod(v.shape) for v in m.trainable_weights))
    assert params==(1231937 if trainable else 31937),(rid,params)
    print("GLOVE_START",rid,"trainable",trainable,"Train",len(y),
          "Val",len(yv),"params",params,"smoke",smoke,flush=True)
    m.summary(print_fn=lambda s:print(s,flush=True))
    epochs=1 if smoke else cfg["common_training"]["max_epochs"]
    start=time.perf_counter()
    history=m.fit(x,y,validation_data=(xv,yv),
          batch_size=cfg["common_training"]["batch_size"],
          epochs=epochs,shuffle=True,verbose=2)
    duration=time.perf_counter()-start
    start=time.perf_counter()
    probabilities=m.predict(xv,batch_size=256,verbose=0).reshape(-1)
    infer=time.perf_counter()-start
    preds,result=score(yv,probabilities)
    hist={k:[float(v) for v in vals] for k,vals in history.history.items()}
    result.update({"val_loss":hist["val_loss"][-1]})
    print("GLOVE_VAL",rid,"accuracy",round(result["val_accuracy"],6),
       "f1",round(result["val_f1"],6),"loss",round(result["val_loss"],6),
       "epochs",len(hist["loss"]),"sec",round(duration,2),flush=True)
    if smoke:
        del m
        tf.keras.backend.clear_session()
        return
    m.save(model_path)
    pred_frame=pd.DataFrame({
        "example_id":val.example_id.to_numpy(),
        "label":yv,
        "prediction":preds,
        "positive_probability":probabilities,
    })
    pred_frame.to_csv(pred_csv,index=False,lineterminator="\n",float_format="%.12g")
    hist_path=R/f"{rid}_history.csv"
    pd.DataFrame(hist).assign(epoch=np.arange(1,len(hist["loss"])+1)).to_csv(
       hist_path,index=False,lineterminator="\n")
    figures(rid,hist,result)
    saved_embedding=m.get_layer("embedding").get_weights()[0]
    change=float(np.linalg.norm(saved_embedding-matrix))
    if trainable:assert change>0.1, "Fine-tuned embedding unchanged?"
    else:assert np.array_equal(saved_embedding,matrix),"Frozen embedding unexpectedly changed"
    record={
      "run_id":rid,
      "condition":"GloVe_100d_Frozen" if not trainable else "GloVe_100d_FineTuning",
      "dataset":"dedup_v1.1",
      "seed":42,
      "scope":"Validation-only exploratory extension after official Gate E",
      "original_baseline":"DL-03-s42",
      "embedding_initialization":"GloVe_6B_100d_Train_only_vocab",
      "embedding_trainable":trainable,
      "trainable_params":params,
      "frozen_params":1231937-params,
      "vocabulary_size":12000,
      "input_length":160,
      "embedding_dim":100,
      "hidden_units":64,
      "epochs_completed":len(hist["loss"]),
      "batch_size":128,
      "optimizer":"Adam_0.001",
      "loss":"binary_crossentropy",
      "dropout":0,
      "early_stopping":False,
      "train_time_sec":round(duration,5),
      "inference_time_sec":round(infer,5),
      "validation_size":4956,
      "test_evaluations":0,
      "glove_source_sha256":provenance["glove_binary_sha256"],
      "glove_initial_weights_sha256":provenance["aligned_matrix_sha256"],
      "glove_vocabulary_coverage":json.loads((R/"glove_coverage.json").read_text())["vocab_exact_match_fraction"],
      "embedding_change_l2":change,
      "model_path":str(model_path.relative_to(ROOT)),
      "model_sha256":file_hash(model_path),
      "predictions_path":str(pred_csv.relative_to(ROOT)),
      "predictions_sha256":file_hash(pred_csv),
      "history_path":str(hist_path.relative_to(ROOT)),
      "history_sha256":file_hash(hist_path),
      "config_sha256":file_hash(CFG),
      "created_at_utc":datetime.now(timezone.utc).isoformat(),
      "hardware":[str(g) for g in dev],
      **result,
    }
    target.write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("GLOVE_RUN_PERSISTED",rid,"model_sha",record["model_sha256"],flush=True)
    del m
    tf.keras.backend.clear_session()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--runs",nargs="+",choices=RUNS,default=RUNS)
    args=parser.parse_args()
    for rid in args.runs:run_one(rid,args.smoke)
    print("GLOVE_OPTIONAL_TRAINING_COMPLETE",args.runs,flush=True)

if __name__=="__main__":
    main()
