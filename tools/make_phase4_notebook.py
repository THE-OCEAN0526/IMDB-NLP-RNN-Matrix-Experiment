"""Create executable 07 results-comparison Notebook; no Test/retraining actions."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/"notebooks/07_IMDB_Results_Comparison.ipynb"
assert not PATH.exists(),"Refuse to overwrite existing Notebook 07"

def M(s):return nbf.v4.new_markdown_cell(s)
def C(s):return nbf.v4.new_code_cell(s)

cells=[
 M("""# 07｜IMDB NLP 矩陣實驗：14 組結果總比較與模型定案

**Phase 4 — Validation-only / Test 前版本。**

本 Notebook 完成規格書要求的五項研究問題 RQ1～RQ5、14 組排名、F1／Accuracy／Precision／Recall、成本與參數量、Loss 曲線、Confusion Matrix，以及模型最終選擇紀錄。

- 資料版本：**dedup_v1.1**，Train 19,823／Validation 4,956。
- 共 14 組：ML-01～04、DL-01～10；GloVe 是選做，不列入。
- 選模指標：**Validation F1**，同分比較 Accuracy，再比較 Run ID。
- **官方 Test 尚未使用。** 本 Notebook 不進行模型訓練，也不計算 Test 分數。
- 模型、Validation 預測與選模鎖定資料皆已保存，重新執行本 Notebook 只讀取既有結果。
"""),
 C("""from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT=Path('/workspace/imdb_matrix_experiment/variants/dedup_v1_1')
assert ROOT.is_dir()
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.common import file_hash
from src.phase4_compare import ALIASES, load_and_check, research_questions
policy,manifest,runs=load_and_check()
lock=json.loads((ROOT/'results/model_selection_lock.json').read_text())
assert lock['status']=='LOCKED_BEFORE_FINAL_TEST'
assert lock['official_test_not_evaluated'] is True
assert lock['runs_registry_sha256']==file_hash(ROOT/'results/runs.csv')
assert lock['policy_sha256']==file_hash(ROOT/'config/phase4_selection_policy.yaml')
print('Dataset:',manifest['version'])
print('Runs:',len(runs))
print('Model selection:',lock['selected_overall'],lock['selected_deep_learning'])
print('Test status: NOT EVALUATED')
"""),
 M("""## 1. 14 組模型的統一結果表

F1 是正面類別（label=1）的 binary F1；所有分數都重新根據 Validation prediction CSV 與 Label 計算驗證過。傳統模型沒有 DL 的 trainable_params 欄位，不能填成 0。
"""),
 C("""table=runs[['overall_rank','run_id','display_name','family','val_accuracy','val_precision','val_recall',
            'val_f1','val_loss','trainable_params','epochs_completed','train_time_sec','inference_time_sec']].copy()
display(table.round(4))
assert len(table)==14 and list(table.overall_rank)==list(range(1,15))
print('Best overall:',table.iloc[0]['display_name'],round(table.iloc[0].val_f1,5))
"""),
 M("""## 2. F1 總排名圖：傳統 ML 與深度學習共同呈現

不把 Validation 最佳分數稱作最終 Test 泛化表現。
"""),
 C("""pic=ROOT/'results/figures/07_all_14_validation_f1.png'
assert pic.is_file()
display(Image(filename=str(pic),width=850))
"""),
 M("""## 3. 研究問題 RQ1、RQ2：文字表示與分類器（2×2）

**RQ1** 固定分類器，檢查 BoW ↔ TF-IDF；**RQ2** 固定文字表示，檢查 KNN ↔ SVC。
KNN 使用 cosine brute，SVC 使用原生 sklearn.svm.SVC(kernel='linear')。
"""),
 C("""rq=json.loads((ROOT/'results/phase4_research_questions.json').read_text())
display(pd.DataFrame([
    {'RQ':group,'comparison':k,'Validation F1 difference (percentage points)':v}
    for group,values in rq.items() if group.startswith('RQ')
    for k,v in values.items()
]).round(4))
display(Image(filename=str(ROOT/'results/figures/07_traditional_2x2_f1.png'),width=580))
"""),
 M("""## 4. RQ3：SimpleRNN、LSTM、GRU

三組使用相同 160-token 序列、100-dim Embedding、64 hidden units、10 epochs、無 Dropout／EarlyStopping；參數量不同。比較 Validation 分數與實測成本。
"""),
 C("""single=runs[runs.run_id.isin(['DL-01-s42','DL-02-s42','DL-03-s42'])].sort_values('run_id')
display(single[['run_id','display_name','val_accuracy','val_f1','val_loss','trainable_params',
                'train_time_sec','inference_time_sec']].round(4))
for rid in ('DL-01-s42','DL-02-s42','DL-03-s42'):
    print(rid,ALIASES[rid])
    display(Image(filename=str(ROOT/f'results/figures/{rid}_learning_curve.png'),width=460))
"""),
 M("""## 5. RQ4：LSTM Dropout × Early Stopping

2×2 包含同一份原始 LSTM 基準 DL-02（沒有重訓）。
Early Stopping 使用 val_loss、patience=3、restore_best_weights=True。標註真實訓練 Epoch，不拿不同 Epoch 的實驗當成等成本比較。
"""),
 C("""four=['DL-02-s42','DL-04-s42','DL-05-s42','DL-06-s42']
lstm=runs.set_index('run_id').loc[four].reset_index()
display(lstm[['run_id','display_name','val_accuracy','val_f1','val_loss',
              'epochs_completed','trainable_params','train_time_sec']].round(4))
display(Image(filename=str(ROOT/'results/figures/07_regularization_2x2_f1.png'),width=650))
for rid in ('DL-02-s42','DL-05-s42'):
    print(rid,'Loss curve')
    display(Image(filename=str(ROOT/f'results/figures/{rid}_learning_curve.png'),width=540))
"""),
 M("""## 6. RQ5：堆疊、混合及雙向網路

Stacked LSTM、LSTM+GRU、Bidirectional LSTM、Bidirectional GRU 均不使用 Dropout 或 Early Stopping，訓練 10 epochs。比較與相應單層基準的差異；並列模型參數量以免錯認為等參數量實驗。
"""),
 C("""adv=['DL-02-s42','DL-03-s42','DL-07-s42','DL-08-s42','DL-09-s42','DL-10-s42']
advanced=runs.set_index('run_id').loc[adv].reset_index()
display(advanced[['run_id','display_name','val_accuracy','val_f1',
                 'trainable_params','train_time_sec']].round(4))
display(Image(filename=str(ROOT/'results/figures/07_advanced_rnn_f1.png'),width=900))
"""),
 M("""## 7. 效能－成本關係（僅比較深度學習組）

這張散佈圖只納入 DL 模型，X 軸為包含各 Epoch Validation 的訓練耗時。
**傳統 ML 的 fit 時間定義不同**：KNN 距離查詢通常屬於推論、SVC 訓練秒數含兩組 C 候選；不能將它們直接與 DL fit 秒數視為同一種成本。
"""),
 C("""display(Image(filename=str(ROOT/'results/figures/07_deep_learning_cost_f1.png'),width=770))
"""),
 M("""## 8. 代表性 Confusion Matrix：兩個類別最佳模型

依 Validation F1 選出傳統 ML 代表與深度學習代表；兩張混淆矩陣均取 Validation 且計入 4,956 筆樣本。
"""),
 C("""champions=[lock['selected_traditional_ml'],lock['selected_deep_learning']]
assert champions==['ML-04-s42','DL-05-s42']
for rid in champions:
    print(rid,ALIASES[rid])
    display(Image(filename=str(ROOT/f'results/figures/{rid}_val_confusion_matrix.png'),width=470))
"""),
 M("""## 9. Test 前模型選擇鎖定與來源驗證

鎖定檔含 14 組排名、候選選模政策、原始模型與預測檔的 SHA-256。
只有兩個預先選定的類別代表模型會列入下一階段官方 Test 計畫；**不得看 Test 再改選模型**。
"""),
 C("""assert len(lock['all14_ranking'])==14
assert lock['all14_ranking']==runs.run_id.tolist()
assert lock['official_test_not_evaluated']
for rid,item in lock['source_artifacts'].items():
    assert file_hash(ROOT/item['model_path'])==item['model_sha256'],rid
    assert file_hash(ROOT/item['validation_predictions_path'])==item['validation_predictions_sha256'],rid
assert pd.read_csv(ROOT/'results/runs.csv')[['test_accuracy','test_f1']].isna().all().all()
print('LOCK CHECK PASS: 14/14 saved model and prediction hashes matched')
print('Frozen for future final Test:',lock['future_final_test_scope'])
print('Test evaluation performed in this Notebook: 0')
"""),
 M("""## 10. 研究結論、限制與下一步

**Validation 結論**：TF-IDF + SVC 在全部 14 組中排名第一；LSTM + EarlyStopping 是深度學習組最佳。固定 seed=42 並不等於已證實統計顯著差異。

**重要限制**：超過半數評論長於 160 tokens；官方 Test 內部重複評論仍保留；沒有另外進行語意近似去重；各架構參數量與訓練成本未完整匹配。選做的 GloVe 尚未執行。

**下一階段**：先驗收 Test 前鎖定，再依固定範圍對官方 Test 執行一次獨立評估，完成真正的 Gate E，任何 Test 後修改都要明確標成探索性分析。
""")
]
book=nbf.v4.new_notebook(cells=cells,metadata={
    "kernelspec":{"display_name":"Python 3 (ipykernel)","name":"python3","language":"python"},
    "language_info":{"name":"python"},
})
nbf.write(book,PATH)
print("NOTEBOOK_CREATED",PATH,"code_cells",sum(c.cell_type=="code" for c in book.cells))
