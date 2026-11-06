"""Generate and finalize Gate E evidence after independent Test predictions are saved.

The final stage cannot pass without a separate 12/12 Gate E unittest log.
No model inference or training is performed here.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import nbformat

from src.common import ROOT,file_hash

R=ROOT/"results"
F=R/"final_test"
WIN=["ML-04-s42","DL-05-s42"]

def generate(finalize=False):
    data=json.loads((F/"locked_final_test_metrics.json").read_text(encoding="utf8"))
    lock=json.loads((R/"model_selection_lock.json").read_text(encoding="utf8"))
    manifest=json.loads((F/"final_test_output_manifest.json").read_text(encoding="utf8"))
    pre=json.loads((R/"phase4_pretest_acceptance.json").read_text(encoding="utf8"))
    pf=json.loads((R/"final_test_preflight.json").read_text(encoding="utf8"))
    runs=pd.read_csv(R/"runs.csv")
    assert data["status"]=="OFFICIAL_TEST_EVALUATED_ONCE"
    assert data["scope"]==WIN==lock["future_final_test_scope"]
    assert data["model_evaluations_performed"]==2 and data["official_test_rows"]==25000
    assert not data["test_based_retuning"]
    assert len(runs)==14 and runs.run_id.is_unique
    assert runs[["test_accuracy","test_f1","test_loss"]].isna().all().all()
    assert file_hash(R/"model_selection_lock.json")==pf["model_selection_lock_sha256"]
    assert file_hash(R/"runs.csv")==lock["runs_registry_sha256"]
    assert manifest["metrics_sha256"]==file_hash(F/"locked_final_test_metrics.json")
    for rid in WIN:
        assert data["predictions_and_metrics"][rid]["predictions_sha256"]==file_hash(
            F/data["predictions_and_metrics"][rid]["predictions_csv"])
        assert data["predictions_and_metrics"][rid]["model_sha256"]==lock["source_artifacts"][rid]["model_sha256"]
    backup=ROOT/"notebooks/07_IMDB_Results_Comparison_PRETEST.ipynb"
    assert backup.is_file() and file_hash(backup)==pre["notebook_07"]["sha256"]
    bookpath=ROOT/"notebooks/07_IMDB_Results_Comparison.ipynb"
    book=nbformat.read(bookpath,as_version=4)
    code=[cell for cell in book.cells if cell.cell_type=="code"]
    assert len(code)==13 and all(cell.execution_count is not None for cell in code)
    assert not any(o.get("output_type")=="error" for cell in code for o in cell.get("outputs",[]))

    label={"ML-04-s42":"TF-IDF + SVC","DL-05-s42":"LSTM + EarlyStopping"}
    history=runs.set_index("run_id")
    lines=[
        "# IMDB NLP 矩陣實驗 — Gate E 最終驗收與官方 Test 報告",
        "",
        f"**狀態：{'PASS — Gate E 驗收完成（12/12 測試通過）' if finalize else '待驗證 — Test 評估已執行，待 Gate E 獨立測試'}。**",
        "",
        "## 一、資料及實驗範圍",
        "- 平台：ai-server 上的 cnn-jupyterlab，傳統 SVC 由 CPU、LSTM 使用 NVIDIA RTX 2060 GPU。",
        "- 資料：dedup_v1.1；Train 19,823／Validation 4,956／官方 Test 25,000（負面 12,500，正面 12,500）。",
        "- Phase 1/2 已完成 14 組必做條件，全部事先依 Validation F1 排名。",
        "- 在觀看 Test 之前固定選模：第一指標 Validation F1；平手比較 Validation Accuracy，再依 Run ID 字典序。",
        "- 僅評估事先鎖定的兩個代表模型，各進行一次官方 Test 預測；沒有重訓、調參、改分類閾值或改選模型。",
        "- Test 內部重複仍依原官方資料保留；不重新抽樣、不另行移除 Test 評論。",
        "",
        "## 二、最終 Test 主要結果",
        "| 類型 | Run | 模型 | Validation Accuracy | Validation F1 | Test Accuracy | Test Precision | Test Recall | Test F1 | Val→Test F1 變化（百分點） |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family,rid in (("傳統 ML","ML-04-s42"),("深度學習","DL-05-s42")):
        v=history.loc[rid]
        m=data["predictions_and_metrics"][rid]
        delta=100*(m["f1"]-v.val_f1)
        lines.append(
            f"| {family} | {rid} | {label[rid]} | {v.val_accuracy:.2%} | {v.val_f1:.2%} | "
            f"{m['accuracy']:.2%} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} | {delta:+.2f} |"
        )
    lines += [
        "",
        "**觀察：** TF-IDF + SVC 的官方 Test F1 高於 LSTM + EarlyStopping。"
        "兩者的 Test F1 都較已定案的 Validation F1 略低；此為已選模型的獨立驗證，未以 Test 再挑模型。",
        "",
        "## 三、逐筆 Test 結果與混淆矩陣",
        "",
    ]
    for rid in WIN:
        m=data["predictions_and_metrics"][rid]
        lines += [
            f"### {rid} — {label[rid]}",
            f"- Test 筆數：{m['test_size']:,}；推論時間：{m['inference_time_sec']:.2f} 秒。",
            f"- Test Confusion Matrix：{m['confusion_matrix']}，列為真實 Negative／Positive，欄為預測 Negative／Positive。",
            f"- 已存逐筆預測：results/final_test/{m['predictions_csv']}。",
            f"- 預測檔 SHA-256：{m['predictions_sha256']}。",
            f"- 模型 SHA-256：{m['model_sha256']}。",
            f"- 預測規則：{m['prediction_policy']}。",
        ]
        if m["test_loss"] is None:
            lines.append("- Test BCE：不適用（SVC 不具有同定義的 BCE）。")
        else:
            lines.append(f"- Test BCE：{m['test_loss']:.6f}；計算方式：{m['test_loss_definition']}。")
        lines.append("")
    lines += [
        "正式 Test Confusion Matrix：",
        "- results/final_test/ML-04-s42_test_confusion_matrix.png",
        "- results/final_test/DL-05-s42_test_confusion_matrix.png",
        "- results/final_test/finalists_validation_vs_test.png：預先鎖定代表的 Val/Test F1 比較。",
        "",
        "## 四、RQ1～RQ5 對應的 Validation 研究問題",
        "原規格要求 RQ1～RQ5 用 14 組 Validation 成績比較控制變因，完整數字在 results/PHASE4_PRETEST_REPORT.md 與 Notebook 07。",
        "官方 Test 僅用於兩個已選模型的泛化驗證，並未另外評估 14 組 Test 成績以重新排名。",
        "",
        "## 五、可重現性證據",
        "- Test 前模型選擇鎖：results/model_selection_lock.json。",
        f"- 鎖定檔 SHA-256：{file_hash(R/'model_selection_lock.json')}。",
        "- Test 前整體排名：results/phase4_ranked_validation.csv。",
        "- Test 前驗收：results/phase4_pretest_acceptance.json。",
        "- 本次一次性 Test 及輸出雜湊：results/final_test/final_test_output_manifest.json。",
        "- 官方 Test 逐筆預測、指標：results/final_test/locked_final_test_metrics.json。",
        f"- 官方 Test 原始來源 SHA-256：{data['test_source_sha256']}。",
        "- Test 完整原始資料列數 25,000，不更動既有 data/raw/test.parquet 與 data/processed/test.parquet。",
        "- 14 個已訓練模型 SHA-256 與凍結紀錄相符，原 runs.csv 的 14 組 Validation Run 保持不變，Test 欄仍為空白。",
        "- Notebook 07 的 Test 前版本已另存 notebooks/07_IMDB_Results_Comparison_PRETEST.ipynb，保存原始 SHA-256；正式 Notebook 07 另新增 Test 成果與驗算。",
        "",
        "## 六、限制與研究倫理",
        "- 單一 seed=42，缺乏跨種子標準差與信賴區間，因此不能推論兩者在所有條件下的統計顯著差距。",
        "- 序列最大長度 160，使超過半數評論遭截斷；這可能影響深度學習表現。",
        "- 去重只針對不同資料集間完全相同的清理後全文；尚未檢查語意相近評論，官方 Test 自身重複仍保留。",
        "- 進階架構使用相同 hidden units 但不等參數量。",
        "- SVC 的 Phase 1 train_time_sec 含多個候選 C 的 fit；DL fit 秒數包含 Epoch Validation；兩者計時定義不同。",
        "- 正式 Test 不能用來重新調參／重選模型；如後續進行額外探索性 Test 比較，必須清楚與本輪鎖定驗收區分。",
        "",
        "## 七、驗收狀態",
        f"- Gate E：{'PASS，12/12 自動測試通過' if finalize else '等待獨立測試結果'}。",
        "- 使用固定選模規則在查看 Test 前決定代表模型；兩份預測均保留完整 sample IDs、標籤、預測類別、分類分數。",
        "- 所有 Test 指標均可從儲存的逐筆 CSV 重新計算，不依賴原模型重跑。",
        "- GloVe 是原規格選做，未執行；必做 14 組已完成。"
    ]
    (R/"GATE_E_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    acceptance={
       "stage":"Gate E / final official Test",
       "status":"PASS" if finalize else "PENDING_RECOMPUTATION_TESTS",
       "timestamp_utc":datetime.now(timezone.utc).isoformat(),
       "dataset":"dedup_v1.1",
       "test_rows":25000,
       "official_test_model_count":2,
       "test_retuning":False,
       "models_evaluated":WIN,
       "gate_a_status":"PASS","gate_b_status":"PASS","gate_c_status":"PASS",
       "phase4_pretest_status":"PASS_PRETEST_LOCKED",
       "phase4_pretest_lock_sha256":file_hash(R/"model_selection_lock.json"),
       "test_metrics_sha256":file_hash(F/"locked_final_test_metrics.json"),
       "validation_registry_sha256":file_hash(R/"runs.csv"),
       "original_pretest_notebook_sha256":file_hash(backup),
       "executed_final_notebook07_sha256":file_hash(bookpath),
       "notebook07_code_cells_executed":len(code),
       "test_metrics":{
          rid:{key:m[key] for key in ("accuracy","precision","recall","f1","confusion_matrix")}
          for rid,m in data["predictions_and_metrics"].items()
       },
       "gate_e_unittest":{
          "passed":12 if finalize else 0,"failed":0 if finalize else None,
          "log_sha256":file_hash(R/"gate_e_unittest.log") if finalize else None
       },
       "report":"results/GATE_E_REPORT.md",
       "test_report":"results/final_test/locked_final_test_metrics.json",
       "caution":"Tests are a one-time, pre-selected scope of two family-best models, not all 14 models.",
    }
    if finalize:
        l=(R/"gate_e_unittest.log").read_text(encoding="utf8")
        assert "Ran 12 tests in" in l and "\nOK\n" in l
    (R/"gate_e_acceptance.json").write_text(json.dumps(acceptance,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("GATE_E_REPORT_"+("FINAL_PASS" if finalize else "PREPARED"),R/"GATE_E_REPORT.md")
    print("GATE_E_ACCEPTANCE",acceptance["status"],"Test_eval_count",2)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--finalize",action="store_true")
    generate(args.parse_args().finalize)
