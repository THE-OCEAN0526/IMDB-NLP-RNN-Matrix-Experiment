# IMDB NLP 矩陣實驗 Phase 4 — Test 前選模驗收

**狀態：PASS_PRETEST_LOCKED。Gate E 正式 Test 尚未進行，Gate E 不可標記 PASS。**

## 完成項目
- 照原始規格書建立必做 Notebook 07：14 組 Validation 排名、RQ1～RQ5、統一指標與模型成本分析。
- Notebook 07：10/10 程式儲存格已在 ai-server JupyterLab 執行，錯誤 0。
- 5 張新的研究問題與比較圖；引用既有 Loss curves 與代表性 Validation Confusion Matrix。
- 14 份正式 Validation predictions CSV 與 14 個模型，全部驗證 SHA-256。
- Test 前選模鎖定：results/model_selection_lock.json，來源和模型的 hash 已固定。
- 12/12 自動驗收：results/phase4_pretest_unittest.log。

## 鎖定模型
- 全體與傳統 ML 最佳：ML-04-s42（TF-IDF + SVC），Validation F1 89.32%。
- 深度學習最佳：DL-05-s42（LSTM + EarlyStopping），Validation F1 84.52%。
- 下一階段建議僅這兩個事先選定的模型進行一次官方 Test 獨立評估。

## 版本與完整性
- Original Gate A/B/C 成果與已訓練模型未修改；Gate B 七個原模型也重新通過雜湊檢查。
- 官方 Test 沒有執行任何模型預測；results/runs.csv 的 Test 指標欄位全部空白。
- 選做 GloVe（Notebook 06）未執行，不影響 14 個必做條件完成狀態。

## 文件
- results/PHASE4_PRETEST_REPORT.md：14 組總表、5 個研究問題的數值分析、評估限制。
- results/phase4_ranked_validation.csv：排序及完整 Validation 數據。
- results/phase4_research_questions.json：RQ1～RQ5 計算結果。
- results/FINAL_TEST_PROTOCOL.md：Test 前模型與範圍鎖定、操作與驗收守則。
- results/phase4_pretest_acceptance.json：機器可讀驗收資訊。

## 下一個步驟
**執行事先鎖定的兩個代表模型的一次性官方 Test 評估，之後產出 Gate E 最終驗收與包含 Test 的最終 Notebook 07 版本。**
本輪仍保持 Test 未被評估，故不能報告 Test Accuracy／Test F1。
