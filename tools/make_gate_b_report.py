"""Produce formal Phase 1 / Gate B report from 7 completed Validation-only runs."""
from __future__ import annotations
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import ROOT,file_hash
from src.evaluation import phase1_settings

RUNS=["ML-01-s42","ML-02-s42","ML-03-s42","ML-04-s42",
      "DL-01-s42","DL-02-s42","DL-03-s42"]

def make_report():
    cfg,cfg_sha,manifest=phase1_settings()
    records=pd.read_csv(ROOT/"results/runs.csv")
    selected=records.set_index("run_id").loc[RUNS].reset_index()
    assert len(selected)==7 and records.run_id.nunique()==7
    assert all(selected.status=="validation_complete")
    assert all(pd.isna(selected[c]).all() for c in
          ("test_accuracy","test_precision","test_recall","test_f1","test_loss"))
    assert all(selected.split_hash==manifest["split_assignments_sha256"])
    fig_dir=ROOT/"results/figures"
    fig_dir.mkdir(parents=True,exist_ok=True)
    names=selected.run_id.str.replace("-s42","",regex=False)
    fig,ax=plt.subplots(figsize=(9,4.4),dpi=160)
    bars=ax.bar(names,selected.val_f1.to_numpy())
    ax.set_ylim(0,1);ax.set_xlabel("Run ID");ax.set_ylabel("Validation F1")
    ax.set_title("IMDB dedup_v1.1 — Phase 1 baseline comparison (Validation only)")
    ax.grid(axis="y",alpha=.2)
    for bar,v in zip(bars,selected.val_f1):
        ax.annotate(f"{v:.3f}",(bar.get_x()+bar.get_width()/2,v),
                    ha="center",va="bottom",fontsize=8,xytext=(0,3),textcoords="offset points")
    fig.tight_layout()
    chart=fig_dir/"phase1_validation_f1_comparison.png"
    fig.savefig(chart)
    plt.close(fig)
    # Do not present SVC candidate-search time as equivalent to one DL model's 10-epoch fit.
    fig,ax=plt.subplots(figsize=(8,4.3),dpi=160)
    separate=selected.sort_values("val_f1",ascending=False)
    bar_labels=separate.run_id.str.replace("-s42","",regex=False)
    ax.barh(bar_labels,separate.val_accuracy.to_numpy())
    ax.set_xlim(0,1);ax.invert_yaxis()
    ax.set_xlabel("Validation accuracy")
    ax.set_title("IMDB dedup_v1.1 — held-out Validation accuracy")
    ax.grid(axis="x",alpha=.2)
    fig.tight_layout()
    accuracy_chart=fig_dir/"phase1_validation_accuracy_comparison.png"
    fig.savefig(accuracy_chart);plt.close(fig)

    gate=ROOT/"results/gate_b_unittest.log"
    assert gate.is_file() and "Ran 10 tests in" in gate.read_text()
    assert "\nOK\n" in gate.read_text()
    notebooks={}
    for name in ("02_IMDB_Traditional_ML_Matrix.ipynb","03_IMDB_RNN_Baselines.ipynb"):
        doc=json.loads((ROOT/"notebooks"/name).read_text(encoding="utf-8"))
        code=[c for c in doc["cells"] if c["cell_type"]=="code"]
        errors=[o.get("ename","") for c in code for o in c.get("outputs",[]) if o.get("output_type")=="error"]
        assert code and all(c["execution_count"] is not None for c in code) and not errors
        notebooks[name]={"code_cells":len(code),"all_executed":True,"saved_errors":0,
                         "sha256":file_hash(ROOT/"notebooks"/name)}
    result={
        "phase":"1","gate":"B","status":"PASS",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "dataset":"dedup_v1.1",
        "train_records":19823,"validation_records":4956,"test_records_evaluated":0,
        "phase1_config_sha256":cfg_sha,
        "phase0_config_sha256":manifest["config_sha256"],
        "split_assignments_sha256":manifest["split_assignments_sha256"],
        "run_count":7,"run_ids":RUNS,
        "gate_b_tests":{"passed":10,"failed":0,"log_sha256":file_hash(gate)},
        "notebooks":notebooks,
        "f1_plot":str(chart.relative_to(ROOT)),
        "accuracy_plot":str(accuracy_chart.relative_to(ROOT)),
        "results":[
          {"run_id":r.run_id,"val_accuracy":round(float(r.val_accuracy),8),
           "val_precision":round(float(r.val_precision),8),
           "val_recall":round(float(r.val_recall),8),
           "val_f1":round(float(r.val_f1),8),
           "trainable_params":None if pd.isna(r.trainable_params) else int(r.trainable_params),
           "train_time_sec":float(r.train_time_sec),
           "inference_time_sec":float(r.inference_time_sec),
           "model_path":str(r.model_path),"predictions_path":str(r.predictions_path)}
          for _,r in selected.iterrows()
        ],
        "warnings":[
          "Gate B records only Validation results; official Test sealed until Gate E.",
          "Model selection / tuning on the same Validation set can favor selected parameters; final Test must not be used for repeated tuning.",
          "Two SVC candidate training times are summed; KNN training time excludes the expensive nearest-neighbor Validation query.",
          "DL baseline uses all 10 epochs without EarlyStopping or Dropout; last-epoch performance can be worse than the best Validation epoch.",
          "Input truncation: over half the original reviews exceed 160 tokens.",
          "GPU seed 42 does not guarantee perfect determinism across hardware and TensorFlow kernels.",
        ],
    }
    jpath=ROOT/"results/gate_b_acceptance.json"
    jpath.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    markdown=[
      "# IMDB NLP 矩陣實驗 — Phase 1 / Gate B 基準模型驗收報告",
      "",
      "驗收：**PASS — 7 組正式基準，Validation 證據與模型完整，10/10 測試通過。**",
      "",
      "## 1. 研究資料與公平性",
      "- 執行環境：ai-server / Docker cnn-jupyterlab；KNN/SVC 使用 CPU，DL 使用 GPU RTX 2060。",
      "- 資料版本：dedup_v1.1；Train 19,823，Validation 4,956；官方 Test 25,000 筆**未參與本階段**。",
      "- 隨機種子：42，Train-only BoW / TF-IDF / Keras TextVectorization；相同清理規則 phase0-text-v1。",
      "- Gate A 分割 Hash："+manifest["split_assignments_sha256"],
      "- Phase 1 凍結設定 Hash："+cfg_sha,
      "",
      "## 2. 七組 Validation 結果",
      "| Run ID | 類別 | 模型 | Accuracy | Precision | Recall | F1 | 訓練秒數 | 推論秒數 | Trainable params |",
      "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _,r in selected.iterrows():
        typ="傳統 ML" if r.run_id.startswith("ML") else "深度學習"
        label=("BoW " if r.representation=="bow" else "TF-IDF ") + ("KNN" if "KNeighbors" in r.model_name else "SVC") if typ=="傳統 ML" else r.model_name
        params="—" if pd.isna(r.trainable_params) else f"{int(r.trainable_params):,}"
        markdown.append(f"| {r.run_id} | {typ} | {label} | {r.val_accuracy*100:.2f}% | {r.val_precision*100:.2f}% | {r.val_recall*100:.2f}% | {r.val_f1*100:.2f}% | {r.train_time_sec:.2f} | {r.inference_time_sec:.2f} | {params} |")
    markdown += [
       "",
       "所有分數均為同一份 Validation 的實測結果，不是測試集分數。",
       "",
       "## 3. 傳統 ML 超參數選擇（只用 Validation F1）",
       "KNN：metric=cosine、algorithm=brute、k=3/5；SVC：**原生 sklearn.svm.SVC**、kernel=linear、C=0.1/1.0。",
       "| Run | Selected k 或 C | 候選及 F1 |",
       "|---|---|---|",
    ]
    for rid in RUNS[:4]:
        x=json.loads((ROOT/f"results/{rid}_candidate_selection.json").read_text())
        pairs=", ".join(f"{row['value']}: {row['val_f1']:.4f}" for row in x["candidates"])
        markdown.append(f"| {rid} | {x['chosen_parameter']} | {pairs} |")
    markdown += [
       "",
       "## 4. 單層循環網路架構",
       "三組共用 Embedding(12000, 100)、hidden units=64、Adam(0.001)、BCE、batch=128、epochs=10、無 Dropout、無 EarlyStopping。",
       "| Run | Parameters | Epochs | 初始 Val Loss | 最終 Val Loss |",
       "|---|---:|---:|---:|---:|",
    ]
    for _,r in selected[selected.run_id.str.startswith("DL")].iterrows():
        hist=pd.read_csv(ROOT/r.history_path)
        markdown.append(f"| {r.run_id} | {int(r.trainable_params):,} | {int(r.epochs_completed)} | {hist.val_loss.iloc[0]:.4f} | {hist.val_loss.iloc[-1]:.4f} |")
    markdown += [
       "",
       "深度學習中的訓練時間包含每一 Epoch 的 Validation；SVC 訓練時間合計兩組 C 候選的 fit，兩種時間不應當作相同定義的單模型成本。",
       "",
       "## 5. Notebook／模型／圖表／預測",
       "- Notebook 02、03：程式儲存格全部實際執行，保存錯誤 0。",
       "- 模型：artifacts/models／(4 個 .joblib + 3 個 .keras)。",
       "- Validation 預測：results/predictions／（7 個 CSV，含逐筆 ID、標籤及預測）。",
       "- 圖表：results/figures／（7 個 Confusion Matrix、3 個 DL Loss 曲線、Phase 1 F1 與 Accuracy 對照圖）。",
       "- 正式結果：results/runs.csv（7 個不同 Run ID，Test 成績欄空白）。",
       "- 10/10 自動測試：results/gate_b_unittest.log。",
       "",
       "## 6. 限制與後續",
       "- Test 未被使用，也未將 Validation 最佳 Epoch 暗中改成 Early Stopping；不以 Test 分數調參。",
       "- 本次為單一 seed=42，尚不提供穩定性的統計顯著性結論。",
       "- 輸入長度 160 截斷約 54–55% 評論，需後續揭露。",
       "- SVC 候選與 KNN 候選皆依相同 Validation 指標選取，不能直接將調參過程的 Validation F1 當作獨立 Test 泛化分數。",
       "",
       "## 7. Gate B 結論",
       "7 組實驗均在 dedup_v1.1 完整資料與固定特徵表示下執行，Validation 指標、模型、參數量、訓練成本、逐筆預測與混淆矩陣可追溯。**Gate B PASS**。",
       "下一階段可進入 Phase 2：Dropout × EarlyStopping 與進階 RNN 架構；LSTM 基準 DL-02 不重複訓練。",
    ]
    (ROOT/"results/GATE_B_REPORT.md").write_text("\n".join(markdown)+"\n",encoding="utf-8")
    print("GATE_B_PASS runs=7 tests=10 notebooks=2 Test_evaluations=0",flush=True)
    print("ACCEPTANCE_FILE",jpath)
    print("REPORT_FILE",ROOT/"results/GATE_B_REPORT.md")
    print(selected[["run_id","val_accuracy","val_f1","train_time_sec"]].to_string(index=False))
if __name__=="__main__":
    make_report()
