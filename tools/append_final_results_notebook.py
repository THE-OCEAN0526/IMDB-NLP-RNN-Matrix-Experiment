"""Preserve exact Pre-Test Notebook 07 and append read-only final Test reporting."""
from __future__ import annotations
import json
import nbformat as nbf
from src.common import ROOT,file_hash

R=ROOT/"results"
ORIGINAL=ROOT/"notebooks/07_IMDB_Results_Comparison.ipynb"
ARCHIVE=ROOT/"notebooks/07_IMDB_Results_Comparison_PRETEST.ipynb"
OUT=R/"final_test/locked_final_test_metrics.json"

def add():
    assert OUT.is_file(),"Official Test has not completed"
    assert not ARCHIVE.exists(),"Pre-Test archive already exists"
    before=json.loads((R/"phase4_pretest_acceptance.json").read_text())
    assert file_hash(ORIGINAL)==before["notebook_07"]["sha256"],"Pre-Test notebook changed"
    final=json.loads(OUT.read_text())
    assert final["status"]=="OFFICIAL_TEST_EVALUATED_ONCE"
    assert final["scope"]==["ML-04-s42","DL-05-s42"]
    with ARCHIVE.open("xb") as f:
        f.write(ORIGINAL.read_bytes())
    assert file_hash(ARCHIVE)==before["notebook_07"]["sha256"]
    nb=nbf.read(ORIGINAL,as_version=4)
    codes=[c for c in nb.cells if c.cell_type=="code"]
    assert len(codes)==10 and all(c.execution_count is not None for c in codes)
    assert not any(o.get("output_type")=="error" for c in codes for o in c.get("outputs",[]))
    nb.cells[0].source=(
        "# 07｜IMDB NLP 矩陣實驗：14 組 Validation 比較＋最終 Test\n\n"
        "**版本區分：**前半部為原先的 Test 前選模歷史快照（14 組 Validation）。"
        "後半部為鎖定模型後的一次性官方 Test 獨立評估結果。"
        "測試前版本已原封保留為 07_IMDB_Results_Comparison_PRETEST.ipynb；"
        "原選模檔 SHA256 不變。Notebook 只讀取保存的 Test 成績與逐筆預測，"
        "**不再訓練或進行新一輪模型推論**。"
    )
    for c in nb.cells:
        if c.cell_type!="code":continue
        c.source=c.source.replace(
            "print('Test status: NOT EVALUATED')",
            "print('Historical status at selection lock: Test NOT EVALUATED')"
        ).replace(
            "print('Test evaluation performed in this Notebook: 0')",
            "print('Model inference performed in this Notebook: 0; read-only output')")
    nb.cells.extend([
        nbf.v4.new_markdown_cell(
           "# 11｜官方 Test：鎖定後獨立評估\n\n"
           "事前依 Validation F1 選定兩個代表：**ML-04-s42** 與 **DL-05-s42**。"
           "每組只對原版 25,000 筆官方 Test 評估一次。"
           "本段**只讀取**既有 final_test 預測檔，沒有再執行模型推論或調參。\n\n"
           "前面提到的 Test 未評估，是選模鎖定當時的歷史狀態。"
        ),
        nbf.v4.new_code_cell(
           "from pathlib import Path\n"
           "import json,numpy as np,pandas as pd\n"
           "from IPython.display import Image,display\n"
           "F=ROOT/'results/final_test'\n"
           "final=json.loads((F/'locked_final_test_metrics.json').read_text(encoding='utf8'))\n"
           "lock_now=json.loads((ROOT/'results/model_selection_lock.json').read_text(encoding='utf8'))\n"
           "assert final['status']=='OFFICIAL_TEST_EVALUATED_ONCE'\n"
           "assert final['scope']==lock_now['future_final_test_scope']==['ML-04-s42','DL-05-s42']\n"
           "assert final['model_selection_lock_sha256']==file_hash(ROOT/'results/model_selection_lock.json')\n"
           "assert final['model_evaluations_performed']==2 and not final['test_based_retuning']\n"
           "print('Official Test: evaluated after selection lock; fixed scope',final['scope'])\n"
           "print('Official Test rows:',final['official_test_rows'])\n"
           "print('New model inferences performed in Notebook 07: zero')"
        ),
        nbf.v4.new_markdown_cell(
           "## 12｜最終 Test 指標與 Validation 對照\n\n"
           "分別呈現 Test Accuracy、Precision、Recall、F1；LSTM 的 BCE 可計算，"
           "但 SVC 無同一定義的 Binary Cross-Entropy，因此不填造 ML Test Loss。"
           "Test 不能拿來重新選模。"
        ),
        nbf.v4.new_code_cell(
           "v=pd.read_csv(ROOT/'results/runs.csv').set_index('run_id')\n"
           "records=[]\n"
           "for rid in final['scope']:\n"
           "    m=final['predictions_and_metrics'][rid]\n"
           "    records.append({'run_id':rid,'model':m['model'],\n"
           "       'Validation Accuracy':v.loc[rid,'val_accuracy'],\n"
           "       'Test Accuracy':m['accuracy'],\n"
           "       'Validation F1':v.loc[rid,'val_f1'],\n"
           "       'Test F1':m['f1'],\n"
           "       'Test Precision':m['precision'],\n"
           "       'Test Recall':m['recall'],\n"
           "       'Test BCE (if applicable)':m.get('test_loss')})\n"
           "display(pd.DataFrame(records).round(4))\n"
           "display(Image(filename=str(F/final['comparison_figure']),width=750))"
        ),
        nbf.v4.new_markdown_cell(
           "## 13｜Test Confusion Matrix 與逐筆數據一致性\n\n"
           "只對已保存的 25,000 筆 Test CSV 重新計算指標，"
           "**沒有重新執行原模型**；比對來源樣本 ID 與標籤後才接受結果。"
        ),
        nbf.v4.new_code_cell(
           "from sklearn.metrics import accuracy_score,f1_score,confusion_matrix\n"
           "canonical=pd.read_parquet(ROOT/'data/processed/test.parquet')\n"
           "for rid in final['scope']:\n"
           "    m=final['predictions_and_metrics'][rid]\n"
           "    file=F/m['predictions_csv']\n"
           "    p=pd.read_csv(file)\n"
           "    assert len(p)==len(canonical)==25000\n"
           "    assert p.example_id.tolist()==canonical.example_id.tolist()\n"
           "    assert p.label.tolist()==canonical.label.tolist()\n"
           "    assert file_hash(file)==m['predictions_sha256']\n"
           "    assert abs(accuracy_score(p.label,p.prediction)-m['accuracy'])<1e-12\n"
           "    assert abs(f1_score(p.label,p.prediction)-m['f1'])<1e-12\n"
           "    assert confusion_matrix(p.label,p.prediction,labels=[0,1]).tolist()==m['confusion_matrix']\n"
           "    print(rid,'Official Test F1',round(m['f1'],6),'CM',m['confusion_matrix'])\n"
           "    display(Image(filename=str(F/f'{rid}_test_confusion_matrix.png'),width=410))\n"
           "print('Recomputed stored Test metrics: PASS')"
        ),
        nbf.v4.new_markdown_cell(
           "## 14｜研究結論與限制\n\n"
           "- RQ1～RQ5 與 14 組模型排名使用 Validation；最終只有鎖定的兩組代表使用 Test。\n"
           "- 單一 seed 42，沒有統計顯著性結論；模型參數量及計時範圍並非完全一致。\n"
           "- 160-token 截斷超過半數評論，官方 Test 內部重複仍保留，"
           "未進行近似語意去重。\n"
           "- 無 Test 後重訓、改參數或改選模型。\n\n"
           "正式結果與 Gate E 驗收報告在 results/final_test 與 results/GATE_E_REPORT.md。"
        )
    ])
    nbf.validate(nb)
    nbf.write(nb,ORIGINAL)
    print("PRETEST_ARCHIVED",ARCHIVE,"sha256",file_hash(ARCHIVE))
    print("FINAL_NOTEBOOK_UPDATED",ORIGINAL,
          "code_cells",len([c for c in nb.cells if c.cell_type=="code"]))
    print("NO_NEW_TEST_MODEL_PREDICTIONS_IN_NOTEBOOK")

if __name__=="__main__":
    add()
