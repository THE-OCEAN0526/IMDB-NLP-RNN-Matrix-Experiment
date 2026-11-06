"""Generate a durable Gate A acceptance record from validated v1.1 artifacts."""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, config, file_hash

cfg, config_hash = config()
m=json.loads((ROOT/"data/processed/split_manifest.json").read_text())
a=json.loads((ROOT/"results/data_audit.json").read_text())
f=json.loads((ROOT/"results/feature_metadata.json").read_text())
log=(ROOT/"results/gate_a_unittest.log").read_text(encoding="utf-8")
assert "Ran 11 tests in" in log and "\nOK\n" in log
assert m["config_sha256"]==f["config_sha256"]==config_hash
assert m["audit"]["train_val_test_cross_split_cleaned_text_overlaps"]=={
    "train_validation":0,"train_test":0,"validation_test":0}
assert m["counts"]["train"]["rows"]==19823 and m["counts"]["validation"]["rows"]==4956
assert m["counts"]["test"]["rows"]==25000
notebooks={}
for path in sorted((ROOT/"notebooks").glob("*.ipynb")):
    nb=nbformat.read(path,as_version=4)
    codes=[c for c in nb.cells if c.cell_type=="code"]
    errors=[o.get("ename","") for c in codes for o in c.get("outputs",[]) if o.get("output_type")=="error"]
    assert codes and all(c.get("execution_count") is not None for c in codes) and not errors
    notebooks[path.name]={"code_cells":len(codes),"executed":len(codes),"errors":errors,"sha256":file_hash(path)}
assert len(notebooks)==2
pre_v1={
   "config/experiment.yaml":"46bbd9516a471e8e08163d02bed3833a0a8ca9982801cbf96e44893554f66e07",
   "data/processed/split_manifest.json":"60373e2185ae57e9ac279b46556c90183d01af3611982c7ff2de79c4af678243",
   "notebooks/00_IMDB_Data_Preparation.ipynb":"05466e515a665ce87ec9a8d3830a35ab32d974c658bb351368d7c0b5765b04fd",
   "notebooks/01_IMDB_Preprocessing_Vectorization.ipynb":"556c58bcc7f1ab02272c647de6bd34673b1c51fa10e8cf611cff1f32395faa0e",
}
legacy=ROOT.parents[1]
assert all(file_hash(legacy/n)==sha for n,sha in pre_v1.items())
report={
  "version":"dedup_v1.1",
  "phase":"0",
  "gate":"A",
  "status":"PASS_EXACT_CLEANED_TEXT_CROSS_SPLIT",
  "verified_at_utc":datetime.now(timezone.utc).isoformat(),
  "test_result":{"passed":11,"failed":0,"exit_code":0,"log_sha256":file_hash(ROOT/"results/gate_a_unittest.log")},
  "notebooks":notebooks,
  "dataset_counts":m["counts"],
  "removed_from_original_train":{
       "overlap_with_official_test":a["excluded_test_overlap"],
       "duplicates_within_remaining_pool":a["excluded_intra_train_duplicates"],
  },
  "cross_split_full_cleaned_text_overlap_counts":a["train_val_test_cross_split_cleaned_text_overlaps"],
  "preserved_test_internal_duplicates":{
      "groups":a["test_internal_duplicate_groups"],
      "extra_rows":a["test_internal_extra_duplicate_rows"],
  },
  "config_sha256":config_hash,
  "split_assignments_sha256":m["split_assignments_sha256"],
  "feature_metadata_sha256":file_hash(ROOT/"results/feature_metadata.json"),
  "source_sha256":m["source_raw_sha256"],
  "preprocessing_version":m["preprocessing_version"],
  "fit_policy":f["fit_policy"],
  "features":f["features"],
  "sequence_length":f["sequence_length"],
  "lengths":f["lengths"],
  "v1_0_unchanged_by_sha256":True,
  "limitations":[
      "Exact full-cleaned-text collisions across splits are zero. Near duplicates and semantic overlaps were not checked.",
      "Official Test internal duplicate reviews deliberately retained to preserve the official benchmark.",
      "Token length 160 truncates over 50% of comments; report this limitation in Phase 1.",
      "No model training or Test-driven model selection performed in Phase 0.",
  ],
}
(ROOT/"results/gate_a_acceptance.json").write_text(
    json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
lines=[
  "# IMDB NLP 矩陣實驗 — v1.1 去重版 Phase 0 / Gate A 驗收報告",
  "",
  "驗收狀態：**PASS（已確認 Train／Validation／Test 間完全相同的清理後完整文字為零）**",
  "",
  "## 1. 檔案位置與版本",
  "- 執行環境：ai-server / Docker cnn-jupyterlab / 自有 JupyterLab",
  "- 專案獨立變體：/home/xuan/cnn-workspace/imdb_matrix_experiment/variants/dedup_v1_1/",
  "- v1.0 所有來源、資料、Notebook、設定、結果保持不變（SHA-256 已比對）",
  "- v1.1 設定 SHA-256："+config_hash,
  "- 切分清單 SHA-256："+m["split_assignments_sha256"],
  "- 原始 Train/Test SHA-256 已核對，官方 Test 25,000 筆及順序完全保持不變",
  "",
  "## 2. 資料切分與篩選",
  "| Split | 筆數 | 負面 0 | 正面 1 | 唯一清理後全文數 |",
  "|---|---:|---:|---:|---:|",
]
for n in ("train","validation","test"):
    c=m["counts"][n]
    lines.append("| "+n+" | "+str(c["rows"])+" | "+str(c["labels"]["0"])+" | "+str(c["labels"]["1"])+" | "+str(c["unique_cleaned_text"])+" |")
lines += [
  "",
  f"- 原 Train 中與官方 Test 相同文字排除：{a['excluded_test_overlap']['rows']} 筆（負面 95／正面 28）",
  f"- 餘下 Train 池內去重：{a['excluded_intra_train_duplicates']['rows']} 筆（負面 69／正面 29）",
  "- 兩份排除清單僅記錄樣本 ID、索引、標籤、Hash 和理由；沒有散佈原文",
  "- 官方 Test 內部重複群組：193，額外重複資料列：201（刻意保留）",
  "",
  "## 3. Gate A 交叉重複與資料洩漏規範",
  "| 原文 hash 比對（完整清理後文字） | 重複組數 |",
  "|---|---:|",
  "| Train / Validation | 0 |",
  "| Train / Test | 0 |",
  "| Validation / Test | 0 |",
  "",
  "- 清理版本：phase0-text-v1（與 v1.0 相同）",
  "- 同一全文 hash 但異標籤群組：0",
  "- 所有 BoW、TF-IDF、Keras TextVectorization fit / adapt 均僅使用新 v1.1 Train；其他 splits 只 transform / inference",
  "- 索引以 source_index 排序，seed=42、stratify=label；輸出 ID 雜湊比對批准的模擬值",
  "- 本結果僅表示已消除完全相同清理後全文的跨組重複，不涵蓋語意近似重複",
  "",
  "## 4. Phase 0 Notebook 和向量化成果",
  "| Notebook | 代碼儲存格 | 執行成功 | 儲存錯誤 |",
  "|---|---:|---:|---:|",
]
for n,x in notebooks.items():
    lines.append(f"| {n} | {x['code_cells']} | {x['executed']} | {len(x['errors'])} |")
lines += [
  "",
  "- BoW 特徵：12,000；TF-IDF 特徵：12,000；Keras 詞彙表：12,000",
  "- Keras 序列長度：160，並儲存 3 個 split 的 token IDs、Labels、Sample IDs",
  "- 6 個 SciPy sparse 特徵矩陣與 2 個已擬合 scikit-learn 向量器皆已保存",
  "",
  "### Token 長度統計",
  "| Split | 超出 160 tokens 的評論 | 比例 |",
  "|---|---:|---:|",
]
for n in ("train","validation","test"):
    r=f["lengths"][n]
    lines.append(f"| {n} | {r['truncated_records']} | {r['truncated_fraction']*100:.2f}% |")
lines += [
  "",
  "## 5. 自動驗收",
  "- Gate A 11/11 項 unittest：**全部通過，Exit Code 0**",
  "- 驗收日誌：results/gate_a_unittest.log",
  "- 機器可讀報告：results/gate_a_acceptance.json",
  "- 已比對 v1.0 兩本 Notebook、原設定及 manifest SHA-256，均未變更",
  "- 曾修正一項測試：TF-IDF CSR 的元素順序不同，但依矩陣索引比較最大差約 6.66e-16，屬浮點誤差；沒有修改向量化結果",
  "",
  "## 6. 仍須揭露的限制",
  "1. 官方 Test 內部原有重複文本仍保留；其測試成績不是去重後的唯一文本測試指標。",
  "2. 未做語意相同／近似重複評論檢查，因此不能宣稱排除所有可能的資料洩漏。",
  "3. 序列長度 160 導致約 54–55% 評論被截斷；後續若調整長度，必須明確建立新實驗條件。",
  "4. 尚未開始 Phase 1 模型訓練，也沒有使用 Test 成績調參。",
  "",
  "## 7. 放行結論",
  "v1.1 的 Phase 0 實作與 Gate A（限定完全相同清理後全文的跨組重複）已通過，可供後續 Phase 1 新模型矩陣使用；"
  "所有模型必須使用此 v1.1 切分與詞彙表重新訓練，不得混用 v1.0 模型成績。",
]
(ROOT/"results/GATE_A_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
(ROOT/"docs/V1.1_APPROVAL_RECORD.md").write_text(
   "# v1.1 去重規格批准與套用紀錄\n\n"
   "使用者於本次對話批准前一份 v1.1 去重規格並指示建立獨立變體及重新驗收 Phase 0 / Gate A。\n\n"
   "批准的審查依據：/home/xuan/cnn-workspace/imdb_matrix_experiment/docs/IMDB_資料去重與切分規格_v1.1_審查草案.md\n\n"
   "套用位置：/home/xuan/cnn-workspace/imdb_matrix_experiment/variants/dedup_v1_1/\n\n"
   "套用結果：Phase 0 兩本 Notebook 已完整執行，Gate A 11/11 項自動測試通過。\n"
   "原審查草案仍保留歷史狀態；此批准紀錄與實際結果以 results/GATE_A_REPORT.md、results/gate_a_acceptance.json 為準。\n",
   encoding="utf-8")
print("WROTE",ROOT/"results/GATE_A_REPORT.md")
print("WROTE",ROOT/"results/gate_a_acceptance.json")
print("WROTE",ROOT/"docs/V1.1_APPROVAL_RECORD.md")
print("GATE_A_PASS 11/11 NO_CROSS_SPLIT_IDENTICAL_FULL_CLEANED_TEXT")
