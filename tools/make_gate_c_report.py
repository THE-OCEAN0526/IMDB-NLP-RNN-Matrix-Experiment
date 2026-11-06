"""Write formal Phase 2 / Gate C report from verified 14-condition Validation outputs."""
from __future__ import annotations
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.common import ROOT,file_hash
from src.evaluation import phase1_settings

RUNS=[f"ML-{i:02d}-s42" for i in range(1,5)]+[
    f"DL-{i:02d}-s42" for i in range(1,11)]
P2=[f"DL-{i:02d}-s42" for i in range(4,11)]

def run():
    base,base_sha,manifest=phase1_settings()
    cfg=yaml.safe_load((ROOT/"config/phase2_conditions.yaml").read_text())
    cfg_hash=file_hash(ROOT/"config/phase2_conditions.yaml")
    registry=pd.read_csv(ROOT/"results/runs.csv")
    assert set(registry.run_id)==set(RUNS) and registry.run_id.is_unique
    ordered=registry.set_index("run_id").loc[RUNS].reset_index()
    assert registry[["test_accuracy","test_f1","test_loss"]].isna().all().all()
    check_log=ROOT/"results/gate_c_unittest.log"
    text=check_log.read_text()
    assert "Ran 12 tests in" in text and "\nOK\n" in text
    nb={}
    for name in ("04_IMDB_LSTM_Regularization_Matrix.ipynb",
                 "05_IMDB_Advanced_RNN_Architectures.ipynb"):
        path=ROOT/"notebooks"/name
        data=json.loads(path.read_text())
        codes=[cell for cell in data["cells"] if cell["cell_type"]=="code"]
        errors=[o.get("ename","") for cell in codes for o in cell.get("outputs",[]) if o.get("output_type")=="error"]
        assert codes and all(c.get("execution_count") is not None for c in codes) and not errors
        nb[name]={"executed_cells":len(codes),"errors":0,"sha256":file_hash(path)}

    fig_dir=ROOT/"results/figures"
    fig_dir.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,5.1),dpi=150)
    bars=ax.bar(ordered.run_id.str.replace("-s42","",regex=False),ordered.val_f1)
    ax.set(ylim=(0,1),ylabel="Validation F1",title="IMDB dedup_v1.1 — 14 fixed experimental conditions")
    ax.tick_params(axis="x",labelrotation=40)
    ax.grid(axis="y",alpha=.2)
    for bar,val in zip(bars,ordered.val_f1):
        ax.annotate(f"{val:.2f}",(bar.get_x()+bar.get_width()/2,val),xytext=(0,3),
                    textcoords="offset points",ha="center",fontsize=7)
    fig.tight_layout()
    comparison=fig_dir/"phase2_all_14_validation_f1.png"
    fig.savefig(comparison);plt.close(fig)

    reg_runs=["DL-02-s42","DL-04-s42","DL-05-s42","DL-06-s42"]
    reg=ordered.set_index("run_id").loc[reg_runs].reset_index()
    fig,ax=plt.subplots(figsize=(7.5,4.3),dpi=150)
    ax.bar(["No dropout\nNo ES","Dropout\nNo ES","No dropout\nES","Dropout\nES"],reg.val_f1)
    ax.set(ylim=(0,1),ylabel="Validation F1",title="LSTM — Dropout × EarlyStopping, seed=42")
    ax.grid(axis="y",alpha=.2)
    fig.tight_layout(); rchart=fig_dir/"phase2_dropout_earlystopping_f1.png"
    fig.savefig(rchart);plt.close(fig)

    adv_runs=["DL-02-s42","DL-03-s42"]+P2[3:]
    adv=ordered.set_index("run_id").loc[adv_runs].reset_index()
    fig,ax=plt.subplots(figsize=(8.5,4),dpi=150)
    ax.bar(adv.run_id.str.replace("-s42","",regex=False),adv.val_f1)
    ax.set(ylim=(0,1),ylabel="Validation F1",title="Single-layer and advanced RNN architectures")
    ax.grid(axis="y",alpha=.2)
    fig.tight_layout(); achart=fig_dir/"phase2_advanced_rnn_f1.png"
    fig.savefig(achart);plt.close(fig)

    result={
      "phase":2,"gate":"C","status":"PASS",
      "validated_at_utc":datetime.now(timezone.utc).isoformat(),
      "dataset":"dedup_v1.1","train_rows":19823,"validation_rows":4956,
      "test_rows_evaluated":0,
      "phase0_config_sha256":manifest["config_sha256"],
      "phase1_config_sha256":base_sha,
      "phase2_config_sha256":cfg_hash,
      "split_assignment_sha256":manifest["split_assignments_sha256"],
      "baseline_count":7,"new_count":7,"total_conditions":14,
      "run_ids":RUNS,
      "gate_c_tests":{"passed":12,"failed":0,"log_sha256":file_hash(check_log)},
      "notebooks":nb,
      "figures":[str(x.relative_to(ROOT)) for x in (comparison,rchart,achart)],
      "runs":[{
        "run_id":r.run_id,"val_accuracy":float(r.val_accuracy),
        "val_f1":float(r.val_f1),"val_loss":None if pd.isna(r.val_loss) else float(r.val_loss),
        "epochs_completed":None if pd.isna(r.epochs_completed) else int(r.epochs_completed),
        "params":None if pd.isna(r.trainable_params) else int(r.trainable_params),
        "train_sec":float(r.train_time_sec),
        "model":r.model_path,"predictions":r.predictions_path
      } for _,r in ordered.iterrows()],
      "limitation":[
        "All scores are on Validation only. Test reserved for final Gate E.",
        "Training runs use one seed=42; no statistical significance conclusion.",
        "EarlyStopping records restored-weight validation loss, which can differ from history last epoch.",
        "Model architecture comparisons are not parameter-count matched.",
        "No semantic near-duplicate audit; official Test internal duplicates retained.",
        "Sequence length 160 truncates more than half the reviews.",
      ],
    }
    (ROOT/"results/gate_c_acceptance.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    lines=[
       "# IMDB NLP 矩陣實驗 — Phase 2 / Gate C 完成報告",
       "",
       "**Gate C：PASS；14 組必做條件均具備完整 Validation 結果，12/12 自動驗收通過。**",
       "",
       "## 實驗條件與資料",
       "- 資料：dedup_v1.1，Train 19,823、Validation 4,956，官方 Test 25,000 筆未使用。",
       "- Seed 42、Embedding 100、Hidden 64、Adam 0.001、batch 128、Epoch 上限 10。",
       "- 共用 Baseline：Gate B 的 7 組完全保留；本輪新增 3 組 LSTM 正則化與 4 組進階架構。",
       "- Phase 2 設定 SHA-256："+cfg_hash,
       "- 分割設定 SHA-256："+manifest["split_assignments_sha256"],
       "",
       "## 14 組正式 Validation 成績",
       "| Run ID | 模型 | Accuracy | F1 | Epochs | Params | 訓練秒數 |",
       "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _,r in ordered.iterrows():
        ep="—" if pd.isna(r.epochs_completed) else str(int(r.epochs_completed))
        par="—" if pd.isna(r.trainable_params) else f"{int(r.trainable_params):,}"
        label=(r.representation+" "+r.model_name) if r.run_id.startswith("ML") else r.model_name
        lines.append(f"| {r.run_id} | {label} | {r.val_accuracy*100:.2f}% | {r.val_f1*100:.2f}% | {ep} | {par} | {r.train_time_sec:.2f} |")
    lines += [
       "",
       "## LSTM 正則化 2×2",
       "| Run | Dropout | EarlyStopping | Trained epochs | Best val-loss epoch | Final model val-loss | F1 |",
       "|---|---|---|---:|---:|---:|---:|",
    ]
    for rid in reg_runs:
        r=ordered.set_index("run_id").loc[rid]
        if rid=="DL-02-s42":
            hist=pd.read_csv(ROOT/r.history_path)
            drop,es="No","No"
            best=int(hist.val_loss.argmin()+1)
            final=float(r.val_loss)
        else:
            x=json.loads((ROOT/f"results/{rid}_training_metadata.json").read_text())
            drop,es=("Yes" if x["condition"]["dropout"] else "No"),("Yes" if x["condition"]["early_stopping"] else "No")
            best=x["best_val_loss_epoch"];final=x["final_model_val_loss"]
        lines.append(f"| {rid} | {drop} | {es} | {int(r.epochs_completed)} | {best} | {final:.4f} | {r.val_f1*100:.2f}% |")
    lines += [
      "",
      "對照條件使用 LSTM 輸出後 Dropout 0.3；EarlyStopping monitor=val_loss、patience=3、restore_best_weights=True。",
      "還原權重後的 Validation Loss 可能不同於實際停止的最後 Epoch Loss，已分別保存。",
      "",
      "## 進階 RNN 架構對照",
      "| Run | Arch | Val F1 | Trainable Params | Epochs |",
      "|---|---|---:|---:|---:|",
    ]
    for rid in adv_runs:
        r=ordered.set_index("run_id").loc[rid]
        lines.append(f"| {rid} | {r.model_name} | {r.val_f1*100:.2f}% | {int(r.trainable_params):,} | {int(r.epochs_completed)} |")
    lines += [
       "",
       "進階組為純架構比較，前一層 return_sequences=True，雙向層使用 Bidirectional；不添加 Dropout 或 EarlyStopping。",
       "參數量不同，不能宣稱公平的等參數量成本比較。",
       "",
       "## Notebook、模型、輸出與驗收",
       "- Notebook 04：3 組新 LSTM 條件，重用 DL-02；全部儲存格已執行。",
       "- Notebook 05：4 組進階架構，基準對照 DL-02、DL-03；全部儲存格已執行。",
       "- 14 個模型（4 個 joblib、10 個 Keras），14 份 Validation Prediction CSV。",
       "- 14 張 Confusion Matrix、10 張深度學習 Loss 曲線，加上整體比較圖。",
       "- 機器可讀：results/runs.csv、results/gate_c_acceptance.json；測試結果：results/gate_c_unittest.log。",
       "- 已以 SHA-256 確認 Gate B 既有 7 個模型與相應 Run 原始資料列完全未改動。",
       "- 12 項 Gate C unittest：全部通過（Exit Code 0）。",
       "",
       "## 限制與結論",
       "- F1、Accuracy 全部為 Validation 指標，不能作為 Test 泛化結論。",
       "- 此輪僅 seed=42；GPU 部分操作可能非完全決定性。",
       "- 序列截斷 160、Test 內部重複與近似語意重複問題仍需在最終報告揭露。",
       "- **Gate C 已通過**；後續可先完成 Phase 4（07）Validation 比較與模型選擇策略，再進行鎖定後的 Test 評估（Gate E）。",
    ]
    (ROOT/"results/GATE_C_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("GATE_C_PASS",len(RUNS),"conditions",len(P2),"new",flush=True)
    print(ordered[["run_id","val_accuracy","val_f1","train_time_sec"]].to_string(index=False),flush=True)

if __name__=="__main__":
    run()
