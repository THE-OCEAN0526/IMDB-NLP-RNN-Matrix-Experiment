# IMDB NLP 矩陣實驗 — Gate E 最終驗收與官方 Test 報告

**狀態：PASS — Gate E 驗收完成（12/12 測試通過）。**

## 一、資料及實驗範圍
- 平台：ai-server 上的 cnn-jupyterlab，傳統 SVC 由 CPU、LSTM 使用 NVIDIA RTX 2060 GPU。
- 資料：dedup_v1.1；Train 19,823／Validation 4,956／官方 Test 25,000（負面 12,500，正面 12,500）。
- Phase 1/2 已完成 14 組必做條件，全部事先依 Validation F1 排名。
- 在觀看 Test 之前固定選模：第一指標 Validation F1；平手比較 Validation Accuracy，再依 Run ID 字典序。
- 僅評估事先鎖定的兩個代表模型，各進行一次官方 Test 預測；沒有重訓、調參、改分類閾值或改選模型。
- Test 內部重複仍依原官方資料保留；不重新抽樣、不另行移除 Test 評論。

## 二、最終 Test 主要結果
| 類型 | Run | 模型 | Validation Accuracy | Validation F1 | Test Accuracy | Test Precision | Test Recall | Test F1 | Val→Test F1 變化（百分點） |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 傳統 ML | ML-04-s42 | TF-IDF + SVC | 89.21% | 89.32% | 88.10% | 88.50% | 87.58% | 88.03% | -1.29 |
| 深度學習 | DL-05-s42 | LSTM + EarlyStopping | 84.54% | 84.52% | 83.44% | 84.64% | 81.70% | 83.15% | -1.37 |

**觀察：** TF-IDF + SVC 的官方 Test F1 高於 LSTM + EarlyStopping。兩者的 Test F1 都較已定案的 Validation F1 略低；此為已選模型的獨立驗證，未以 Test 再挑模型。

## 三、逐筆 Test 結果與混淆矩陣

### ML-04-s42 — TF-IDF + SVC
- Test 筆數：25,000；推論時間：239.22 秒。
- Test Confusion Matrix：[[11077, 1423], [1553, 10947]]，列為真實 Negative／Positive，欄為預測 Negative／Positive。
- 已存逐筆預測：results/final_test/ML-04-s42_test_predictions.csv。
- 預測檔 SHA-256：8f21b91d85791dc6575750993fb95596936ee903d7eea5dac879ba97e552eafe。
- 模型 SHA-256：bab9ec9083d5160fa2c58bc8b96e134764bcc6682854d3ecb53a956fa9ff353d。
- 預測規則：Original sklearn.svm.SVC.predict, no new threshold。
- Test BCE：不適用（SVC 不具有同定義的 BCE）。

### DL-05-s42 — LSTM + EarlyStopping
- Test 筆數：25,000；推論時間：1.04 秒。
- Test Confusion Matrix：[[10647, 1853], [2287, 10213]]，列為真實 Negative／Positive，欄為預測 Negative／Positive。
- 已存逐筆預測：results/final_test/DL-05-s42_test_predictions.csv。
- 預測檔 SHA-256：0bdc4c4f8ec6fd613079ab773cd264188d301364717a2f03cc83106470d8a27d。
- 模型 SHA-256：a7b02a17c31b690e13d2bc59c62824bf61fc6a8b5df7db0587904a3d87aaf1e3。
- 預測規則：Fixed model sigmoid score >= 0.5。
- Test BCE：0.387129；計算方式：sklearn.log_loss on fixed Keras sigmoid output (binary cross-entropy)。

正式 Test Confusion Matrix：
- results/final_test/ML-04-s42_test_confusion_matrix.png
- results/final_test/DL-05-s42_test_confusion_matrix.png
- results/final_test/finalists_validation_vs_test.png：預先鎖定代表的 Val/Test F1 比較。

## 四、RQ1～RQ5 對應的 Validation 研究問題
原規格要求 RQ1～RQ5 用 14 組 Validation 成績比較控制變因，完整數字在 results/PHASE4_PRETEST_REPORT.md 與 Notebook 07。
官方 Test 僅用於兩個已選模型的泛化驗證，並未另外評估 14 組 Test 成績以重新排名。

## 五、可重現性證據
- Test 前模型選擇鎖：results/model_selection_lock.json。
- 鎖定檔 SHA-256：6313bacff701e5aedae6239cf25d5b57cacaeeda761a3ff1102a3f3e59bf59c1。
- Test 前整體排名：results/phase4_ranked_validation.csv。
- Test 前驗收：results/phase4_pretest_acceptance.json。
- 本次一次性 Test 及輸出雜湊：results/final_test/final_test_output_manifest.json。
- 官方 Test 逐筆預測、指標：results/final_test/locked_final_test_metrics.json。
- 官方 Test 原始來源 SHA-256：b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da。
- Test 完整原始資料列數 25,000，不更動既有 data/raw/test.parquet 與 data/processed/test.parquet。
- 14 個已訓練模型 SHA-256 與凍結紀錄相符，原 runs.csv 的 14 組 Validation Run 保持不變，Test 欄仍為空白。
- Notebook 07 的 Test 前版本已另存 notebooks/07_IMDB_Results_Comparison_PRETEST.ipynb，保存原始 SHA-256；正式 Notebook 07 另新增 Test 成果與驗算。

## 六、限制與研究倫理
- 單一 seed=42，缺乏跨種子標準差與信賴區間，因此不能推論兩者在所有條件下的統計顯著差距。
- 序列最大長度 160，使超過半數評論遭截斷；這可能影響深度學習表現。
- 去重只針對不同資料集間完全相同的清理後全文；尚未檢查語意相近評論，官方 Test 自身重複仍保留。
- 進階架構使用相同 hidden units 但不等參數量。
- SVC 的 Phase 1 train_time_sec 含多個候選 C 的 fit；DL fit 秒數包含 Epoch Validation；兩者計時定義不同。
- 正式 Test 不能用來重新調參／重選模型；如後續進行額外探索性 Test 比較，必須清楚與本輪鎖定驗收區分。

## 七、驗收狀態
- Gate E：PASS，12/12 自動測試通過。
- 使用固定選模規則在查看 Test 前決定代表模型；兩份預測均保留完整 sample IDs、標籤、預測類別、分類分數。
- 所有 Test 指標均可從儲存的逐筆 CSV 重新計算，不依賴原模型重跑。
- GloVe 是原規格選做，未執行；必做 14 組已完成。

## 八、最終跨階段回歸測試
- Gate A 11/11、Gate B 10/10、Gate C 12/12、Gate E 12/12；**總計 45/45 通過**。
- 詳細日誌：results/final_regression_A_B_C_E.log。
- 歷史 Gate B 測試改為檢查七個基準 Run 均存在且沒有重複，不再錯誤要求當前 Registry 只能有 7 個 Run；未更動模型、向量器或分數。
