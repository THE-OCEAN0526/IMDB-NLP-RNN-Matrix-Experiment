"""Produce GloVe optional Validation comparison, independent of locked Gate E."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from src.common import ROOT,file_hash

OPT=ROOT/"optional_glove"
R=OPT/"results"
RUNS=["TL-01-s42","TL-02-s42"]

def generate():
    cfg=yaml.safe_load((OPT/"experiment.yaml").read_text())
    assert cfg["comparison"]["official_Test_evaluation"]=="prohibited"
    original=pd.read_csv(ROOT/"results/runs.csv")
    base=original.loc[original.run_id=="DL-03-s42"]
    assert len(original)==14 and len(base)==1
    b=base.iloc[0]
    new={rid:json.loads((R/f"{rid}.json").read_text()) for rid in RUNS}
    assert all(r["test_evaluations"]==0 for r in new.values())
    for r in new.values():
        assert file_hash(ROOT/r["model_path"])==r["model_sha256"]
        assert file_hash(ROOT/r["predictions_path"])==r["predictions_sha256"]
    rows=[
        {
          "run_id":"DL-03-s42","embedding_initialization":"Random Trainable",
          "embedding_trainable":True,"model":"GRU(64)","dataset":"dedup_v1.1",
          "val_accuracy":float(b.val_accuracy),"val_precision":float(b.val_precision),
          "val_recall":float(b.val_recall),"val_f1":float(b.val_f1),
          "val_loss":float(b.val_loss),
          "trainable_params":int(b.trainable_params),"epochs_completed":int(b.epochs_completed),
          "train_time_sec":float(b.train_time_sec),
          "source":"Existing independent Phase1/GateB baseline; no re-training",
        }
    ]
    for rid in RUNS:
        r=new[rid]
        rows.append({
         "run_id":rid,"embedding_initialization":r["condition"],
         "embedding_trainable":r["embedding_trainable"],
         "model":"GRU(64)","dataset":r["dataset"],
         "val_accuracy":r["val_accuracy"],"val_precision":r["val_precision"],
         "val_recall":r["val_recall"],"val_f1":r["val_f1"],
         "val_loss":r["val_loss"],
         "trainable_params":r["trainable_params"],
         "epochs_completed":r["epochs_completed"],
         "train_time_sec":r["train_time_sec"],
         "source":"New post Gate E exploratory GloVe Validation-only run",
        })
    table=pd.DataFrame(rows)
    assert len(table)==3
    csv=R/"three_gru_validation_comparison.csv"
    table.to_csv(csv,index=False,float_format="%.12g",lineterminator="\n")
    labels=["Random\nDL-03","GloVe Frozen\nTL-01","GloVe Fine-tune\nTL-02"]
    for metric,title,basename in [
        ("val_f1","Validation F1 (positive class)","glove_3way_validation_f1.png"),
        ("val_accuracy","Validation Accuracy","glove_3way_validation_accuracy.png"),
    ]:
        fig,ax=plt.subplots(figsize=(7.5,4.6),dpi=160)
        rects=ax.bar(labels,table[metric].to_numpy())
        ax.set_ylim(0,1);ax.set_ylabel(title)
        ax.set_title("IMDB — GRU(64) Embedding initialization comparison")
        ax.grid(axis="y",alpha=.2)
        for rect,v in zip(rects,table[metric]):
            ax.annotate(f"{v:.3f}",(rect.get_x()+rect.get_width()/2,v),
                  xytext=(0,3),textcoords="offset points",ha="center")
        fig.tight_layout()
        fig.savefig(R/basename);plt.close(fig)
    cov=json.loads((R/"glove_coverage.json").read_text())
    src=json.loads((R/"glove_source_manifest.json").read_text())
    frozen=json.loads((R/"FROZEN_OFFICIAL_GATE_E_BEFORE_OPTIONAL.json").read_text())
    assert all(file_hash(ROOT/p)==sha for p,sha in frozen["source_artifacts_sha256"].items())
    delta_1=100*(table.iloc[1].val_f1-table.iloc[0].val_f1)
    delta_2=100*(table.iloc[2].val_f1-table.iloc[0].val_f1)
    delta_ft=100*(table.iloc[2].val_f1-table.iloc[1].val_f1)
    h1=pd.read_csv(ROOT/new["TL-01-s42"]["history_path"])
    h2=pd.read_csv(ROOT/new["TL-02-s42"]["history_path"])
    lines=[
       "# IMDB 矩陣實驗 — GloVe 100d 選做實作（補做）",
       "",
       "**狀態：兩組已實際訓練並保存；此為 Gate E 完成後的探索性 Validation 擴充，不重新使用官方 Test。**",
       "",
       "## 1. 研究問題與實驗設計",
       "- RQ6：GloVe 是否能改善 GRU 的分類表現？比較原 DL-03 (Random, trainable) 與新增 TL-01 (GloVe Frozen)、TL-02 (GloVe Fine-tuning)。",
       "- 固定 dedup_v1.1；Train 19,823／Validation 4,956；原始 Train-only Keras TextVectorization，12,000 vocabulary；長度 160。",
       "- Embedding 100 維、GRU 64 units、sigmoid、Adam(0.001)、BCE、batch 128、seed=42、10 Epoch、無 Dropout／EarlyStopping。",
       "- TL-01 只有 31,937 個參數可訓練；TL-02 以及原基準 DL-03 均有 1,231,937 個參數可訓練。",
       "- 沒有重新訓練 DL-03，沒有改變原始 14 組結果或已鎖定的 Gate E 測試。",
       "",
       "## 2. GloVe 來源與詞彙對齊",
       "- Stanford GloVe 6B：Wikipedia 2014 + Gigaword 5，400,000 詞，100 維、uncased。",
       "- 來源網站："+src["canonical_glove_project"],
       "- 本次下載鏡像："+src["download_mirror"],
       "- **格式註記：Zenodo 供應的是將 GloVe 向量轉存的 word2vec float32 binary，並非 Stanford 原始純文字檔。**",
       "- 100d 壓縮包 ZIP MD5："+src["zip_md5"],
       "- 二進位向量 SHA-256："+src["glove_binary_sha256"],
       "- 對齊矩陣 SHA-256："+src["aligned_matrix_sha256"],
       "- 授權："+src["license"],
       f"- Train-only vocabulary 的 11,998 個可匹配詞中，精確匹配 {cov['exact_matching_vocab']:,} 個（{100*cov['vocab_exact_match_fraction']:.2f}%）。",
       f"- 依 Train 已知 token 次數加權，GloVe 精確匹配覆蓋率 {100*cov['train_coverage_excluding_UNK']:.2f}%。",
       "- PAD=0 對應零向量並啟用 mask_zero=True；UNK=1 及未命中詞使用 seed=42 的 N(0,0.05) 初始化。",
       "- Validation／Test 均未參與詞彙表 fit 或向量對齊；拼字或標點附著使部分詞無法與 GloVe 精確對應。",
       "",
       "## 3. 三種 Embedding × GRU 的 Validation 實際成績",
       "| Run | Embedding | Trainable Params | Epoch | Val Accuracy | Val Precision | Val Recall | Val F1 | Val Loss | Fit 秒 |",
       "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _,r in table.iterrows():
        lines.append(f"| {r.run_id} | {r.embedding_initialization} | {int(r.trainable_params):,} | {int(r.epochs_completed)} | {r.val_accuracy:.2%} | {r.val_precision:.2%} | {r.val_recall:.2%} | {r.val_f1:.2%} | {r.val_loss:.4f} | {r.train_time_sec:.2f} |")
    lines += [
       "",
       f"- TL-01 Frozen 相對 DL-03 Random：{delta_1:+.2f} Validation F1 百分點。",
       f"- TL-02 Fine-tuning 相對 DL-03 Random：{delta_2:+.2f} Validation F1 百分點。",
       f"- TL-02 Fine-tuning 相對 TL-01 Frozen：{delta_ft:+.2f} Validation F1 百分點。",
       "",
       "## 4. 訓練 Loss 與過度擬合",
       f"- TL-01：最佳 Validation Loss 出現在第 {int(h1.val_loss.argmin()+1)} 個 Epoch；最終 Val Loss {float(h1.val_loss.iloc[-1]):.4f}。",
       f"- TL-02：最佳 Validation Loss 出現在第 {int(h2.val_loss.argmin()+1)} 個 Epoch；最終 Val Loss {float(h2.val_loss.iloc[-1]):.4f}。",
       f"- TL-02 第 10 Epoch Train Accuracy {float(h2.accuracy.iloc[-1])*100:.2f}% ，Validation Accuracy {float(h2.val_accuracy.iloc[-1])*100:.2f}% ，可見過度擬合跡象。",
       "- 本組為架構控制實驗，因此刻意不使用 EarlyStopping；沒有擷取較佳早期權重作為另外一個未宣告模型。",
       "",
       "## 5. 產物與可追溯性",
       "- Notebook：notebooks/06_IMDB_GloVe_Transfer_Learning.ipynb。",
       "- Source：optional_glove/source/glove.6B.100d.zip、optional_glove/source/glove.6B.100d。",
       "- 對齊矩陣、配對 mask：optional_glove/artifacts/。",
       "- TL-01 與 TL-02 的 Keras 訓練模型、10 Epoch history、Validation 逐筆預測與 Confusion Matrix 分別保留。",
       "- 來源、參數及 GPU 執行日誌：optional_glove/results/ 和 optional_glove/preparation.log。",
       "- 三組比較表：optional_glove/results/three_gru_validation_comparison.csv。",
       "- 圖表：optional_glove/results/glove_3way_validation_f1.png、glove_3way_validation_accuracy.png。",
       "",
       "## 6. 研究結論與限制",
       "- 在本次固定 10 Epoch、單次 seed=42 的 Validation 條件下，使用 GloVe 並非必然優於原本隨機初始化 Embedding 的 GRU。",
       "- GloVe 是外部預訓練詞向量，不代表 GRU 分類器已被預訓練；Fine-tuning 和 Frozen 也有不同可訓練參數量。",
       "- 此選做實驗為原 Gate E **完成後**的探索性研究，不能把先前官方 Test 的分數用來重新選擇最優模型。",
       "- 沒有對 TL-01 / TL-02 進行任何官方 Test 推論，也沒有更改原先 Test 前鎖定的 ML-04／DL-05 或 Test 25,000 筆評估紀錄。",
       "- 此輪沒有多種子統計檢定，也不應宣稱差異有統計顯著性。",
    ]
    (R/"OPTIONAL_GLOVE_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    machine={
        "stage":"optional GloVe 100d, after Gate E",
        "data":"dedup_v1.1","test_evaluations":0,
        "source_sha256":src["glove_binary_sha256"],
        "aligned_sha256":src["aligned_matrix_sha256"],
        "compared_runs":["DL-03-s42","TL-01-s42","TL-02-s42"],
        "new_models":["TL-01-s42","TL-02-s42"],
        "comparison_csv_sha256":file_hash(csv),
        "metrics":[{k:(v.item() if hasattr(v,"item") else v) for k,v in row.items()}
                  for row in table.to_dict("records")],
        "delta_f1_pp":{"frozen_minus_random":float(delta_1),
           "finetune_minus_random":float(delta_2),"finetune_minus_frozen":float(delta_ft)},
        "original_frozen_file_count":len(frozen["source_artifacts_sha256"]),
    }
    (R/"optional_glove_comparison.json").write_text(json.dumps(machine,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("OPTIONAL_GLOVE_COMPARISON_READY",flush=True)
    print(table[["run_id","val_accuracy","val_f1","val_loss","trainable_params"]].to_string(index=False),flush=True)
    print("DELTA_PP",machine["delta_f1_pp"],flush=True)

if __name__=="__main__":
    generate()
