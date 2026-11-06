"""Finalize optional Gate D acceptance after successful 57/57 cross-gate tests."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path

import nbformat
import pandas as pd

from src.common import ROOT,file_hash

OPT=ROOT/"optional_glove"
R=OPT/"results"
CASES=["TL-01-s42","TL-02-s42"]

def main():
    log=(R/"gate_d_unittest.log").read_text()
    full=(R/"full_regression_ABCD_E.log").read_text()
    assert "Ran 12 tests in" in log and "\nOK\n" in log
    assert "Ran 57 tests in" in full and "\nOK\n" in full
    c=json.loads((R/"glove_coverage.json").read_text())
    source=json.loads((R/"glove_source_manifest.json").read_text())
    snap=json.loads((R/"FROZEN_OFFICIAL_GATE_E_BEFORE_OPTIONAL.json").read_text())
    report=R/"OPTIONAL_GLOVE_REPORT.md"
    assert report.is_file()
    runs={rid:json.loads((R/f"{rid}.json").read_text()) for rid in CASES}
    baseline=pd.read_csv(ROOT/"results/runs.csv").set_index("run_id").loc["DL-03-s42"]
    assert len(snap["source_artifacts_sha256"])==26
    assert all(file_hash(ROOT/f)==sha for f,sha in snap["source_artifacts_sha256"].items())
    assert all(r["test_evaluations"]==0 for r in runs.values())
    notebook=ROOT/"notebooks/06_IMDB_GloVe_Transfer_Learning.ipynb"
    nb=nbformat.read(notebook,as_version=4)
    code=[cell for cell in nb.cells if cell.cell_type=="code"]
    bad=[o for cell in code for o in cell.get("outputs",[]) if o.get("output_type")=="error"]
    assert len(code)==7 and all(cell.execution_count is not None for cell in code) and not bad
    baseline_f1=float(baseline.val_f1)
    metrics={
      "DL-03-s42":{"name":"Random Embedding (baseline)", "val_f1":baseline_f1,
                   "val_accuracy":float(baseline.val_accuracy),
                   "trainable_params":int(baseline.trainable_params)},
      "TL-01-s42":{"name":"GloVe Frozen", "val_f1":runs["TL-01-s42"]["val_f1"],
                   "val_accuracy":runs["TL-01-s42"]["val_accuracy"],
                   "trainable_params":runs["TL-01-s42"]["trainable_params"]},
      "TL-02-s42":{"name":"GloVe Fine-tuning", "val_f1":runs["TL-02-s42"]["val_f1"],
                   "val_accuracy":runs["TL-02-s42"]["val_accuracy"],
                   "trainable_params":runs["TL-02-s42"]["trainable_params"]},
    }
    record={
       "stage":"Optional Phase 3 / Gate D, post Gate E exploratory GloVe",
       "status":"PASS_POST_GATE_E_VALIDATION_ONLY",
       "original_gate_e_status":"PASS_UNCHANGED",
       "timestamp_utc":datetime.now(timezone.utc).isoformat(),
       "dataset_variant":"dedup_v1.1",
       "original_required_conditions":14,
       "new_optional_conditions":2,
       "total_stored_conditions_across_separate_registries":16,
       "baseline_reused":"DL-03-s42",
       "test_evaluations_performed":0,
       "comparison_data":"Validation only (4956 samples)",
       "new_runs":CASES,
       "glove_source_format":source["source_format"],
       "glove_source_sha256":source["glove_binary_sha256"],
       "glove_source_zip_md5":source["zip_md5"],
       "aligned_matrix_sha256":source["aligned_matrix_sha256"],
       "vocab_exact_match_fraction":c["vocab_exact_match_fraction"],
       "train_token_weighted_coverage_excluding_UNK":c["train_coverage_excluding_UNK"],
       "original_gate_e_snapshots_unchanged":len(snap["source_artifacts_sha256"]),
       "independent_tests":{"passed":12,"failed":0,
          "log":"optional_glove/results/gate_d_unittest.log",
          "log_sha256":file_hash(R/"gate_d_unittest.log")},
       "full_regression":{"passed":57,"failed":0,
          "log":"optional_glove/results/full_regression_ABCD_E.log",
          "log_sha256":file_hash(R/"full_regression_ABCD_E.log")},
       "notebook06":{"path":"notebooks/06_IMDB_GloVe_Transfer_Learning.ipynb",
           "executed_code_cells":7,"errors":0,"sha256":file_hash(notebook)},
       "validation_metrics":metrics,
       "report":"optional_glove/results/OPTIONAL_GLOVE_REPORT.md",
       "report_sha256":file_hash(report),
       "original_test_unchanged":"Test results from locked prior Gate E remain the ONLY official independent Test.",
       "note":"Optional GloVe was added after official Test; must not select it using already exposed Test results.",
    }
    target=R/"gate_d_acceptance.json"
    assert not target.exists(),"Refuse to overwrite previous Gate D acceptance"
    with target.open("x",encoding="utf8") as f:
        f.write(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    md=[
      "# Optional Gate D — GloVe 100 維擴充驗收",
      "",
      "**狀態：PASS_POST_GATE_E_VALIDATION_ONLY。**",
      "",
      "完成原規格矩陣 E 的兩個選做條件：TL-01 Frozen 與 TL-02 Fine-tuning，並重用 DL-03 原基準。",
      "三組均使用相同 Train／Validation 切分、固定 GRU(64) 架構、Embedding 100 維、10 Epoch、Seed 42。",
      "",
      "## Validation 比較（固定第 10 個 Epoch）",
      "| Run | 初始化 | F1 | Accuracy | 可訓練參數 |",
      "|---|---|---:|---:|---:|",
    ]
    for rid in ["DL-03-s42"]+CASES:
        x=metrics[rid]
        md.append(f"| {rid} | {x['name']} | {x['val_f1']:.2%} | {x['val_accuracy']:.2%} | {x['trainable_params']:,} |")
    md += [
      "",
      "## 驗收及交付",
      "- GloVe 6B／100d 來源 ZIP 及二進位向量檔已驗證，含 400,000 個詞及固定雜湊。",
      f"- 11,998 個可比對詞彙中精確匹配 {c['exact_matching_vocab']:,} 個（{c['vocab_exact_match_fraction']:.2%}）；Train 已知 token 加權匹配率 {c['train_coverage_excluding_UNK']:.2%}。",
      "- Notebook 06 完整執行 7/7 程式儲存格、零錯誤；兩個模型各 10 Epoch 完整訓練並儲存。",
      "- Validation 各 4,956 筆預測、Confusion Matrix 和 F1 指標重算完全一致。",
      "- TL-01 Embedding 在訓練後完全保持初始 GloVe 值；TL-02 Embedding 確認已更新。",
      "- 12/12 項選做 Gate D 測試 PASS。",
      "- 所有 Gate A／B／C／D／E 跨階段測試 **57/57 通過**。",
      f"- 原本 Gate E 的 {len(snap['source_artifacts_sha256'])} 個關鍵原始檔雜湊一致，沒有重新做 Test。",
      "",
      "## 結論及重要限制",
      "- 本輪固定第 10 Epoch 的 Validation F1：DL-03 Random 81.02%、TL-01 Frozen 79.99%、TL-02 Fine-tuning 80.34%。",
      "- TL-02 的 Train/Validation Loss 顯示過度擬合，這是原始固定 10 Epoch 設計的一項觀察，不事後改挑更好 Epoch。",
      "- 資料來源為 Zenodo 上的 GloVe 6B／100d word2vec binary 轉存版（不是 Stanford 原始 txt）；已記錄來源與格式。",
      "- GloVe 選做實驗發生於已鎖定並完成的官方 Gate E 後，因此只能稱為探索性 Validation 比較；不得用先前已看過的 Test 重新選模型。",
      "- 不聲稱跨 Seed 的顯著性或適用於所有 NLP 任務。",
      "",
      "詳細訓練歷程、SHA-256、文字向量匹配情況及圖表：optional_glove/results/OPTIONAL_GLOVE_REPORT.md。",
    ]
    (R/"GATE_D_REPORT.md").write_text("\n".join(md)+"\n",encoding="utf8")
    print("OPTIONAL_GATE_D_PASS tests12/12 full_regression57/57 baseline+2=3")
    print("GLOVE_NOTEBOOK06_EXECUTED",len(code),"code cells")
    print("GATE_E_FROZEN_FILES_VERIFIED",len(snap["source_artifacts_sha256"]))
    print("GATE_D_REPORT",R/"GATE_D_REPORT.md")

if __name__=="__main__":
    main()
