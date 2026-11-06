# IMDB 矩陣實驗 Phase 0 — dedup_v1.1

此為原 v1.0 獨立的 v1.1 資料變體，經使用者批准依照原審查草案重做 Phase 0。**不修改 v1.0 的 Notebook、設定、向量器、資料或結果**。

## 資料規則

沿用既有相同的 IMDB 原始 Train/Test（各 25,000 筆）與清理版 phase0-text-v1。先剔除官方 Train 中 123 筆與官方 Test 清理後完整文字相同的評論，再對餘下 24,877 筆依原始索引升冪排序，保留每個 hash 的第一筆，剔除 98 筆。以種子 42、stratify label 將剩下 24,779 筆切為 Train 19,823、Validation 4,956。**Test 25,000 筆不修改**（Test 內部重複仍保留）。

本版本的分割屬新的實驗版本，不可沿用 v1.0 的詞彙表、IDF、Token IDs 或模型分數。

## 執行

使用 cnn-jupyterlab 容器，專案路徑 /workspace/imdb_matrix_experiment/variants/dedup_v1_1。

依序執行 notebooks/00_IMDB_Data_Preparation.ipynb 與 notebooks/01_IMDB_Preprocessing_Vectorization.ipynb。每本 Notebook 均可從固定檔案重新載入設定，不依賴另一 Notebook 記憶體變數。

執行驗收：

python -B -m unittest discover -s tests -p 'test_gate_a.py' -v

檢視 results/GATE_A_REPORT.md 與 data/processed/split_manifest.json。

## 使用限制

Gate A 驗證跨 Train / Validation / Test 的清理後**完整文字**沒有完全重複，不能聲稱不存在近似文字／語意重複。官方 Test 的內部重複保留，因此 Test 不代表唯一文本的準確率評估。160 Token 截斷限制仍須於後續實驗報告揭露。沒有根據 Test 的模型預測、標籤或分數調整模型。

## Phase 1：七個正式基準模型（Gate B PASS）

原規格：4 個 ML 條件（BoW/TF-IDF × KNN/SVC）加 3 個單層循環網路（SimpleRNN/LSTM/GRU）。

### 固定設定
- Phase 1 設定：config/phase1_baselines.yaml。
- 使用 dedup_v1.1 的 Train 19,823 和 Validation 4,956；Test 完全不進行模型評估。
- KNN：cosine、brute、k=3/5，用 Validation F1 選擇；原生 SVC：linear kernel、C=0.1/1.0，同樣依 Validation F1 選擇。
- 深度學習：Embedding(12,000,100)、RNN/LSTM/GRU 64 units、Adam(0.001)、BCE、batch 128、10 epoch、無 Dropout、無 EarlyStopping。
- 固定 seed 42；TensorFlow 訓練使用容器中的 NVIDIA RTX 2060。

### Notebook
- notebooks/02_IMDB_Traditional_ML_Matrix.ipynb：ML-01～ML-04；已實際執行。
- notebooks/03_IMDB_RNN_Baselines.ipynb：DL-01～DL-03；已實際執行。

### 實際成果
- results/runs.csv：7 條 Validation-only 正式 Run。
- results/GATE_B_REPORT.md：完整表格、候選調參過程、模型與成本比較、限制。
- results/gate_b_acceptance.json：機器可讀放行結果。
- results/gate_b_unittest.log：10/10 自動驗收通過。
- results/predictions：7 組逐筆 Validation 預測（ID、label、prediction、model score）。
- results/figures：7 組 confusion matrix、3 張 DL loss 曲線、Phase 1 F1/Accuracy 比較圖。
- artifacts/models：4 個 .joblib、3 個 .keras。

### 重現與驗收
於容器的 /workspace/imdb_matrix_experiment/variants/dedup_v1_1 目錄：

python -B -m unittest discover -s tests -p test_gate_b.py -v

Notebook 已寫為可重新從儲存檔案載入結果的形式；完成的 Run 不會因為重新開啟 Notebook 而重訓。若要改變實驗參數，請建立新設定版本及新 Run ID，避免覆蓋已完成資料。

**Gate B 已通過。** 下一階段為 Phase 2 LSTM 正則化與進階 RNN，比較時重用 DL-02 的正式基準。

## Phase 2：LSTM 正則化與進階循環架構（Gate C PASS）

- 獨立凍結設定：config/phase2_conditions.yaml。不得修改已完成的 Phase 0 / Phase 1 配置或模型。
- Notebook 04：notebooks/04_IMDB_LSTM_Regularization_Matrix.ipynb，LSTM 的 Dropout 0.3 × EarlyStopping（val_loss, patience=3, restore_best_weights=True）2×2，重用 DL-02。
- Notebook 05：notebooks/05_IMDB_Advanced_RNN_Architectures.ipynb，Stacked LSTM、LSTM→GRU、BiLSTM、BiGRU（DL-07～DL-10）。
- 新增 DL-04～DL-10 共 7 個條件，總數 14 個；均保存完整 Validation 指標、模型、epoch history、sample-level predictions、圖表及獨立設定 hash。
- 全程使用 ai-server 原始 JupyterLab 容器可見 GPU RTX 2060；不使用 Colab、不讀取 Test 做調參。
- results/GATE_C_REPORT.md：正式報告與 14 組 Validation 對照表。
- results/gate_c_acceptance.json：機器可讀的 Gate C PASS 驗收。
- results/gate_c_unittest.log：12/12 項自動驗收通過。
- results/phase2_gate_b_frozen_before.json：Phase 2 開始前 Gate B 的 7 個模型及結果行 hash；Gate C 驗證未改動。

### 驗收方式

於 /workspace/imdb_matrix_experiment/variants/dedup_v1_1 執行：

python -B -m unittest discover -s tests -p test_gate_c.py -v

### 後續限制

即使 Gate C 通過，目前仍只有 Validation 成績，不能稱為最終 Test Accuracy。後續應先決定 Validation-only 模型選擇策略，再使用官方 Test 做一次獨立的最終評估。160-token 長度導致大量評論截斷；正式報告應提及。單一 seed=42 不足以判定模型差距的統計顯著性。

## Phase 4：07 結果比較與 Test 前模型選擇（Pre-Test PASS）

- Notebook 07：notebooks/07_IMDB_Results_Comparison.ipynb。已在 ai-server JupyterLab 實際執行 10/10 code cells；直接讀取 14 組已保存的 Validation 結果，沒有重訓，也沒有評估官方 Test。
- config/phase4_selection_policy.yaml：選模預先定義規則（Validation F1 > Accuracy > Run ID），與 14 組 eligible runs。
- results/phase4_ranked_validation.csv：14 組完整排序及 Accuracy／Precision／Recall／F1／參數量／耗時。
- results/phase4_research_questions.json：RQ1～RQ5 比較差異（F1 百分點）。
- results/figures/07_*.png：新增 5 張正式比較圖。
- results/PHASE4_PRETEST_REPORT.md：研究問題、結果摘要與注意事項。
- results/model_selection_lock.json：固定 14 組排名、模型與逐筆 Validation predictions SHA-256，鎖定傳統 ML 代表 ML-04-s42 及深度學習代表 DL-05-s42。
- results/phase4_pretest_unittest.log：Test 前 12/12 項驗收完成。
- results/phase4_pretest_acceptance.json：狀態 PASS_PRETEST_LOCKED，Gate E 仍 PENDING_OFFICIAL_TEST。
- results/FINAL_TEST_PROTOCOL.md：下一階段僅兩個代表模型的單次官方 Test 評估操作規範。

### 注意

本輪尚未執行官方 Test，沒有 Test Accuracy 或 Test F1；**Gate E 尚未通過**。選做 GloVe notebook 06 不在 14 組必做條件內，目前未執行。未改動 Gate A／B／C、既有 14 個模型及其 Validation 正式紀錄。

## Gate E：官方 Test 一次性獨立評估（PASS）

Phase 4 已在 Test 前以 Validation F1 選出兩個代表模型，並把完整 14 組模型 SHA-256、Validation 預測 SHA-256、固定的排名和選模規則存至 results/model_selection_lock.json。

依 results/FINAL_TEST_PROTOCOL.md 的預定範圍，在 ai-server **各評估一次**官方 IMDB Test 25,000 筆（負面 12,500／正面 12,500），沒有重新訓練，也沒有 Test 後重新選模。

| Run | 模型 | Test Accuracy | Test F1 |
|---|---|---:|---:|
| ML-04-s42 | TF-IDF + SVC | 88.096% | 88.0338% |
| DL-05-s42 | LSTM + EarlyStopping | 83.440% | 83.1474% |

### 交付與驗收

- notebooks/07_IMDB_Results_Comparison.ipynb：正式最終版，13/13 code cells 已執行，包含 14 組 Validation 研究比較及僅兩個代表模型的官方 Test 結果。
- notebooks/07_IMDB_Results_Comparison_PRETEST.ipynb：保留 Test 前完全未修改的原版 07 Notebook，與 results/phase4_pretest_acceptance.json 的 SHA-256 相同。
- results/GATE_E_REPORT.md：正式論文式結果、重要比較、Confusion Matrix 與實驗限制。
- results/gate_e_acceptance.json：Gate E PASS 機器可讀驗收報告。
- results/gate_e_unittest.log：12/12 項 Test 指標與來源驗收。
- results/final_test/locked_final_test_metrics.json：選定模型一次性官方 Test 結果。
- results/final_test/*_test_predictions.csv：兩組各 25,000 筆的原 ID／Label／Prediction／分數。
- results/final_test/*_test_confusion_matrix.png：兩組 Test 混淆矩陣。
- results/final_test/finalists_validation_vs_test.png：兩組 Validation 與 Test F1 比較圖。
- results/final_test/final_test_output_manifest.json：正式結果輸出與 SHA-256。

### 重要原則

正式 Test 不執行二次驗證性模型推論；Gate E 的 12 項測試只從保存的 CSV 重算數值。模型本體、Train-only 特徵向量器、資料來源及 results/runs.csv 的 14 組原 Validation 紀錄都未更改。

**必做 7 本 Notebook、14 組模型與 Gate A／B／C／E 均完成。** 原規格的 GloVe 06 為選做，尚未實作。最終 Test 後不應以這份 Test 調整模型並再次宣稱獨立測試。

## Update: GloVe 選做補充矩陣 E 已實作（Optional Gate D PASS）

此段是前述 Gate E 之後的補充更新。過去記錄「GloVe 未實作」描述的是當時的歷史狀態。

- 已建立並實際執行 notebooks/06_IMDB_GloVe_Transfer_Learning.ipynb：7/7 code cells 完成。
- 來源為 GloVe 6B／100d，Zenodo word2vec binary 鏡像，400,000 個詞（註明為二進位轉存格式，非 Stanford 原始純文字）。
- 已將原本 Train-only TextVectorization 的 12,000 詞彙與 GloVe 對齊；在 11,998 個可比對 token 中匹配 8,637（71.99%），Train 已知 token 加權匹配 90.95%。PAD=0、UNK=1 依規格固定處理。
- 共用 Train 19,823／Validation 4,956；重用原 DL-03 隨機 Embedding 基準，不再訓練；新增 TL-01（Frozen）和 TL-02（Fine-tuning）各使用完整 10 epochs、GRU 64、batch 128、seed=42、Adam 0.001。
- Validation F1：DL-03 81.02%、TL-01 Frozen 79.99%、TL-02 Fine-tuning 80.34%。
- Validation Accuracy：DL-03 80.99%、TL-01 81.28%、TL-02 81.98%。依原先固定的主要指標 F1 比較時，隨機初始化基準仍最高。
- 新模型、預測、Loss、Confusion Matrix、GloVe 來源 hash 獨立放在 optional_glove/。
- 詳細報告：optional_glove/results/OPTIONAL_GLOVE_REPORT.md、optional_glove/results/GATE_D_REPORT.md。
- 機器驗收：optional_glove/results/gate_d_acceptance.json；Gate D 專屬 12/12、完整 Gate A/B/C/D/E 回歸 57/57。
- 原本 Gate E 的 26 個關鍵檔案 SHA-256 完全一致；**這是已完成官方 Test 之後的探索性 Validation 補充，不重新評估或修改 Test。**
- 原 JupyterLab 容器目前 GPU runtime 無法初始化 CUDA，本次另以相同 Docker 映像的臨時 GPU 容器執行正式訓練，沒有重啟 JupyterLab。Notebook 06 的已存輸出可直接開啟；不需要再按 Run All。
