# IMDB NLP 矩陣實驗 — Phase 1 / Gate B 基準模型驗收報告

驗收：**PASS — 7 組正式基準，Validation 證據與模型完整，10/10 測試通過。**

## 1. 研究資料與公平性
- 執行環境：ai-server / Docker cnn-jupyterlab；KNN/SVC 使用 CPU，DL 使用 GPU RTX 2060。
- 資料版本：dedup_v1.1；Train 19,823，Validation 4,956；官方 Test 25,000 筆**未參與本階段**。
- 隨機種子：42，Train-only BoW / TF-IDF / Keras TextVectorization；相同清理規則 phase0-text-v1。
- Gate A 分割 Hash：9e25b0d714d1fb555c8c8a58fa6858c392ec60ecfba554b923553626154174e9
- Phase 1 凍結設定 Hash：5b3e976e16d73ab101abd41f9d022a13d319d221760625f3c6586d1db8d64d29

## 2. 七組 Validation 結果
| Run ID | 類別 | 模型 | Accuracy | Precision | Recall | F1 | 訓練秒數 | 推論秒數 | Trainable params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ML-01-s42 | 傳統 ML | BoW KNN | 64.35% | 63.68% | 67.50% | 65.54% | 0.01 | 2.08 | — |
| ML-02-s42 | 傳統 ML | BoW SVC | 87.67% | 87.71% | 87.75% | 87.73% | 194.53 | 34.29 | — |
| ML-03-s42 | 傳統 ML | TF-IDF KNN | 76.86% | 75.09% | 80.67% | 77.78% | 0.00 | 1.83 | — |
| ML-04-s42 | 傳統 ML | TF-IDF SVC | 89.21% | 88.74% | 89.92% | 89.32% | 274.28 | 48.07 | — |
| DL-01-s42 | 深度學習 | SimpleRNN | 70.50% | 69.77% | 72.80% | 71.25% | 18.84 | 0.43 | 1,210,625 |
| DL-02-s42 | 深度學習 | LSTM | 82.47% | 85.25% | 78.71% | 81.85% | 13.67 | 0.17 | 1,242,305 |
| DL-03-s42 | 深度學習 | GRU | 80.99% | 81.25% | 80.80% | 81.02% | 12.25 | 0.16 | 1,231,937 |

所有分數均為同一份 Validation 的實測結果，不是測試集分數。

## 3. 傳統 ML 超參數選擇（只用 Validation F1）
KNN：metric=cosine、algorithm=brute、k=3/5；SVC：**原生 sklearn.svm.SVC**、kernel=linear、C=0.1/1.0。
| Run | Selected k 或 C | 候選及 F1 |
|---|---|---|
| ML-01-s42 | 5 | 3: 0.6437, 5: 0.6554 |
| ML-02-s42 | 0.1 | 0.1: 0.8773, 1.0: 0.8554 |
| ML-03-s42 | 5 | 3: 0.7649, 5: 0.7778 |
| ML-04-s42 | 1.0 | 0.1: 0.8815, 1.0: 0.8932 |

## 4. 單層循環網路架構
三組共用 Embedding(12000, 100)、hidden units=64、Adam(0.001)、BCE、batch=128、epochs=10、無 Dropout、無 EarlyStopping。
| Run | Parameters | Epochs | 初始 Val Loss | 最終 Val Loss |
|---|---:|---:|---:|---:|
| DL-01-s42 | 1,210,625 | 10 | 0.5928 | 1.1702 |
| DL-02-s42 | 1,242,305 | 10 | 0.3724 | 0.7924 |
| DL-03-s42 | 1,231,937 | 10 | 0.4143 | 0.7745 |

深度學習中的訓練時間包含每一 Epoch 的 Validation；SVC 訓練時間合計兩組 C 候選的 fit，兩種時間不應當作相同定義的單模型成本。

## 5. Notebook／模型／圖表／預測
- Notebook 02、03：程式儲存格全部實際執行，保存錯誤 0。
- 模型：artifacts/models／(4 個 .joblib + 3 個 .keras)。
- Validation 預測：results/predictions／（7 個 CSV，含逐筆 ID、標籤及預測）。
- 圖表：results/figures／（7 個 Confusion Matrix、3 個 DL Loss 曲線、Phase 1 F1 與 Accuracy 對照圖）。
- 正式結果：results/runs.csv（7 個不同 Run ID，Test 成績欄空白）。
- 10/10 自動測試：results/gate_b_unittest.log。

## 6. 限制與後續
- Test 未被使用，也未將 Validation 最佳 Epoch 暗中改成 Early Stopping；不以 Test 分數調參。
- 本次為單一 seed=42，尚不提供穩定性的統計顯著性結論。
- 輸入長度 160 截斷約 54–55% 評論，需後續揭露。
- SVC 候選與 KNN 候選皆依相同 Validation 指標選取，不能直接將調參過程的 Validation F1 當作獨立 Test 泛化分數。

## 7. Gate B 結論
7 組實驗均在 dedup_v1.1 完整資料與固定特徵表示下執行，Validation 指標、模型、參數量、訓練成本、逐筆預測與混淆矩陣可追溯。**Gate B PASS**。
下一階段可進入 Phase 2：Dropout × EarlyStopping 與進階 RNN 架構；LSTM 基準 DL-02 不重複訓練。
