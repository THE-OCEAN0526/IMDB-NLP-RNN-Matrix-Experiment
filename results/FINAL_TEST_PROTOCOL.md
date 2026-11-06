# IMDB NLP 矩陣實驗 — 官方 Test 評估事前鎖定規範

> 狀態：**Test 前已鎖定；尚未執行官方 Test 評估；不是 Gate E 最終報告。**
> 資料：dedup_v1.1（Train 19,823；Validation 4,956；官方 Test 25,000）
> 選模依據：僅 Validation binary F1，平手比 Validation Accuracy，再比 Run ID 字典序
> 選擇鎖：results/model_selection_lock.json

## 1. 已決定的正式比較代表

依 14 組 Validation 分數事先決定，以下兩個模型將在獲批准執行最終 Test 時，各進行**一次**評估：

| 類別 | Run ID | 模型與特徵 | Validation F1 |
|---|---|---|---:|
| 傳統 ML 最佳 | ML-04-s42 | 12,000 維 TF-IDF + 原生線性 SVC（C=1.0） | 0.893235 |
| 深度學習最佳 | DL-05-s42 | Train-only Keras TextVectorization + Embedding(100) + LSTM(64)，Early Stopping 已完成並還原權重 | 0.845190 |

最高分的整體模型就是 ML-04；DL-05 是深度學習類別代表。

### 嚴格限制

1. **不允許**根據 Test 結果改選模型、改阈值、改 C/k、改 Epoch、改 Dropout、改輸入長度、修改去重策略或重訓模型再聲稱獨立 Test。
2. 不改動現有 results/runs.csv 的既有正式 14 行 Validation 執行結果。最終 Test 輸出到獨立 results/final_test/ 新檔案；不得覆蓋模型。
3. Test 只用於一次性預定範圍的評估，若後續想擴大為所有 14 組，必須清楚標記為另一階段報告、固定範圍並揭露已看過的 Test 結果。
4. Test 使用官方 25,000 筆，保留原本 Test 內部重複，不在 Test 評估時再去重或重新抽樣。
5. 不得重新 fit TF-IDF 的 IDF、CountVectorizer 或 adapt TextVectorization；只能使用 v1.1 Train 已擬合的成品。

## 2. Test 階段開始前的檢查

- 重新計算 results/model_selection_lock.json 內所有 14 組 model_path / validation_predictions_path SHA-256。
- 檢查 results/runs.csv SHA-256 與 lock 相同；Gate C 仍為 PASS；selection policy SHA 相同。
- 原始官方 Test SHA-256：b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da。
- Train-only 特徵成品存在並可讀：
  - TF-IDF：artifacts/vectorizers/tfidf_vectorizer.joblib 與已保存的 tfidf_test.npz。
  - Keras：artifacts/vectorizers/keras_vocabulary_12000.txt 與 keras_token_ids_12000_160.npz。
- 尚未存在同版正式 Test result 時才允許執行；若發現既有輸出，停止並先核對，不可當作初次測試再次執行。

## 3. 鎖定後的單次評估程序（本輪尚未執行）

- ML-04：讀入已保存的 tfidf_test.npz，使用 artifacts/models/ML-04-s42.joblib 的已擬合 SVC 決策規則得到 25,000 筆預測。
- DL-05：讀入已保存的 X_test token IDs（160 tokens），載入 artifacts/models/DL-05-s42.keras，使用既定 probability >=0.5 預測正面類別。
- 對每一模型計算 Accuracy、Precision、Recall、F1、2×2 Confusion Matrix，保留逐筆 source_index/example_id、label、prediction、score。
- 若有損失函數值，須明確區分 ML 沒有直接對應 BCE 與 DL 的 BCE，不能填造 ML 的 val_loss/test_loss。
- 記錄模型及 Test 預測的 SHA-256、來源資料版本、實際執行時間與環境。
- 把正式結果輸出到 results/final_test/（新建）並提供獨立不可覆寫的 Test 執行紀錄；所有 Validation 結果不改。
- 製作 Test 前已鎖定模型與最終 Test 成績的明確區分表格，並撰寫最終 Gate E 驗收報告。

## 4. Gate E 的放行條件

- 原先 14 組 Validation 矩陣、模型與圖表完整保留。
- Test 只有上述兩個事先選定的代表，各做一次獨立評估。
- 來源 ID、Label 與預測數量正確，所有 Test 指標可從保存的逐筆 CSV 重算。
- Test 結果發布後不更改選模規則，不因 Test 分數改訓練設定。
- 最終 07 Notebook 與 Gate E 報告同時清楚標記哪些是 Validation、哪些是 Test。
- 明確註明單一 Seed 的限制、160-token 截斷問題、Test 內部重複，以及非相同定義的訓練耗時。

**本輪只保存程序、完成 Test 前模型選擇與鎖定；沒有執行官方 Test。**
