"""Persist Gate E pre-Test readiness record; never mark Gate E final PASS."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
import nbformat
import pandas as pd

from src.common import ROOT,file_hash

R=ROOT/"results"
lock=json.loads((R/"model_selection_lock.json").read_text(encoding="utf8"))
log=(R/"phase4_pretest_unittest.log").read_text(encoding="utf8")
run=pd.read_csv(R/"runs.csv")
nbfile=ROOT/"notebooks/07_IMDB_Results_Comparison.ipynb"
notebook=nbformat.read(nbfile,as_version=4)
codes=[cell for cell in notebook.cells if cell.cell_type=="code"]
errors=[item.get("ename") for cell in codes for item in cell.get("outputs",[]) if item.get("output_type")=="error"]
assert len(codes)==10 and all(cell.execution_count is not None for cell in codes) and not errors
assert "Ran 12 tests in" in log and "\nOK\n" in log
assert len(run)==14 and run.run_id.is_unique
assert run[["test_accuracy","test_precision","test_recall","test_f1","test_loss"]].isna().all().all()
assert lock["status"]=="LOCKED_BEFORE_FINAL_TEST"
assert lock["official_test_not_evaluated"]
assert lock["future_final_test_scope"]==["ML-04-s42","DL-05-s42"]
record={
    "stage":"Phase 4 / Pre-Test Validation comparison and model selection",
    "status":"PASS_PRETEST_LOCKED",
    "gate_e_final_status":"PENDING_OFFICIAL_TEST",
    "timestamp_utc":datetime.now(timezone.utc).isoformat(),
    "dataset":"dedup_v1.1",
    "training_rows":19823,"validation_rows":4956,"official_test_evaluations":0,
    "run_count":14,"notebook_count_mandatory":7,
    "notebook_07":{"path":"notebooks/07_IMDB_Results_Comparison.ipynb",
                   "code_cells":len(codes),"executed":len(codes),"saved_errors":len(errors),
                   "sha256":file_hash(nbfile)},
    "tests":{"passed":12,"failed":0,"log_sha256":file_hash(R/"phase4_pretest_unittest.log")},
    "models_locked":14,
    "selected_finalists":lock["future_final_test_scope"],
    "selection_lock_path":"results/model_selection_lock.json",
    "selection_lock_sha256":file_hash(R/"model_selection_lock.json"),
    "validation_ranked_csv_sha256":file_hash(R/"phase4_ranked_validation.csv"),
    "research_questions_sha256":file_hash(R/"phase4_research_questions.json"),
    "gate_c_acceptance_sha256":file_hash(R/"gate_c_acceptance.json"),
    "report_path":"results/PHASE4_PRETEST_REPORT.md",
    "future_final_test_protocol":"results/FINAL_TEST_PROTOCOL.md",
    "optional_glove_status":"NOT_RUN_AS_OPTIONAL",
}
(R/"phase4_pretest_acceptance.json").write_text(
    json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
text=[
  "# IMDB NLP 矩陣實驗 Phase 4 — Test 前選模驗收",
  "",
  "**狀態：PASS_PRETEST_LOCKED。Gate E 正式 Test 尚未進行，Gate E 不可標記 PASS。**",
  "",
  "## 完成項目",
  "- 照原始規格書建立必做 Notebook 07：14 組 Validation 排名、RQ1～RQ5、統一指標與模型成本分析。",
  "- Notebook 07：10/10 程式儲存格已在 ai-server JupyterLab 執行，錯誤 0。",
  "- 5 張新的研究問題與比較圖；引用既有 Loss curves 與代表性 Validation Confusion Matrix。",
  "- 14 份正式 Validation predictions CSV 與 14 個模型，全部驗證 SHA-256。",
  "- Test 前選模鎖定：results/model_selection_lock.json，來源和模型的 hash 已固定。",
  "- 12/12 自動驗收：results/phase4_pretest_unittest.log。",
  "",
  "## 鎖定模型",
  "- 全體與傳統 ML 最佳：ML-04-s42（TF-IDF + SVC），Validation F1 89.32%。",
  "- 深度學習最佳：DL-05-s42（LSTM + EarlyStopping），Validation F1 84.52%。",
  "- 下一階段建議僅這兩個事先選定的模型進行一次官方 Test 獨立評估。",
  "",
  "## 版本與完整性",
  "- Original Gate A/B/C 成果與已訓練模型未修改；Gate B 七個原模型也重新通過雜湊檢查。",
  "- 官方 Test 沒有執行任何模型預測；results/runs.csv 的 Test 指標欄位全部空白。",
  "- 選做 GloVe（Notebook 06）未執行，不影響 14 個必做條件完成狀態。",
  "",
  "## 文件",
  "- results/PHASE4_PRETEST_REPORT.md：14 組總表、5 個研究問題的數值分析、評估限制。",
  "- results/phase4_ranked_validation.csv：排序及完整 Validation 數據。",
  "- results/phase4_research_questions.json：RQ1～RQ5 計算結果。",
  "- results/FINAL_TEST_PROTOCOL.md：Test 前模型與範圍鎖定、操作與驗收守則。",
  "- results/phase4_pretest_acceptance.json：機器可讀驗收資訊。",
  "",
  "## 下一個步驟",
  "**執行事先鎖定的兩個代表模型的一次性官方 Test 評估，之後產出 Gate E 最終驗收與包含 Test 的最終 Notebook 07 版本。**",
  "本輪仍保持 Test 未被評估，故不能報告 Test Accuracy／Test F1。",
]
(R/"PHASE4_PRETEST_ACCEPTANCE.md").write_text("\n".join(text)+"\n",encoding="utf8")
print("PRETEST_ACCEPTANCE_CREATED",R/"phase4_pretest_acceptance.json")
print("PRETEST_REPORT_CREATED",R/"PHASE4_PRETEST_ACCEPTANCE.md")
print("STATUS",record["status"],"GATE_E",record["gate_e_final_status"])
