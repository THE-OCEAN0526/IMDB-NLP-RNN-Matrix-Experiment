# IMDB NLP 矩陣實驗 — Phase 2 / Gate C 完成報告

**Gate C：PASS；14 組必做條件均具備完整 Validation 結果，12/12 自動驗收通過。**

## 實驗條件與資料
- 資料：dedup_v1.1，Train 19,823、Validation 4,956，官方 Test 25,000 筆未使用。
- Seed 42、Embedding 100、Hidden 64、Adam 0.001、batch 128、Epoch 上限 10。
- 共用 Baseline：Gate B 的 7 組完全保留；本輪新增 3 組 LSTM 正則化與 4 組進階架構。
- Phase 2 設定 SHA-256：a13de8d435a85d888ae9565ed462c35630c88b61f0a6a7692e187e967ae436b4
- 分割設定 SHA-256：9e25b0d714d1fb555c8c8a58fa6858c392ec60ecfba554b923553626154174e9

## 14 組正式 Validation 成績
| Run ID | 模型 | Accuracy | F1 | Epochs | Params | 訓練秒數 |
|---|---|---:|---:|---:|---:|---:|
| ML-01-s42 | bow KNeighborsClassifier | 64.35% | 65.54% | — | — | 0.01 |
| ML-02-s42 | bow SVC | 87.67% | 87.73% | — | — | 194.53 |
| ML-03-s42 | tfidf KNeighborsClassifier | 76.86% | 77.78% | — | — | 0.00 |
| ML-04-s42 | tfidf SVC | 89.21% | 89.32% | — | — | 274.28 |
| DL-01-s42 | SimpleRNN | 70.50% | 71.25% | 10 | 1,210,625 | 18.84 |
| DL-02-s42 | LSTM | 82.47% | 81.85% | 10 | 1,242,305 | 13.67 |
| DL-03-s42 | GRU | 80.99% | 81.02% | 10 | 1,231,937 | 12.25 |
| DL-04-s42 | LSTM + Dropout(0.3) | 79.62% | 79.93% | 10 | 1,242,305 | 24.62 |
| DL-05-s42 | LSTM + EarlyStopping | 84.54% | 84.52% | 4 | 1,242,305 | 9.94 |
| DL-06-s42 | LSTM + Dropout(0.3) + EarlyStopping | 83.66% | 83.79% | 5 | 1,242,305 | 12.31 |
| DL-07-s42 | StackedLSTM | 77.82% | 75.56% | 10 | 1,275,329 | 32.70 |
| DL-08-s42 | LSTM_GRU | 80.67% | 81.17% | 10 | 1,267,265 | 33.59 |
| DL-09-s42 | BidirectionalLSTM | 80.04% | 79.58% | 10 | 1,284,609 | 34.43 |
| DL-10-s42 | BidirectionalGRU | 82.16% | 81.77% | 10 | 1,263,873 | 32.69 |

## LSTM 正則化 2×2
| Run | Dropout | EarlyStopping | Trained epochs | Best val-loss epoch | Final model val-loss | F1 |
|---|---|---|---:|---:|---:|---:|
| DL-02-s42 | No | No | 10 | 1 | 0.7924 | 81.85% |
| DL-04-s42 | Yes | No | 10 | 3 | 0.7951 | 79.93% |
| DL-05-s42 | No | Yes | 4 | 1 | 0.3724 | 84.52% |
| DL-06-s42 | Yes | Yes | 5 | 2 | 0.4065 | 83.79% |

對照條件使用 LSTM 輸出後 Dropout 0.3；EarlyStopping monitor=val_loss、patience=3、restore_best_weights=True。
還原權重後的 Validation Loss 可能不同於實際停止的最後 Epoch Loss，已分別保存。

## 進階 RNN 架構對照
| Run | Arch | Val F1 | Trainable Params | Epochs |
|---|---|---:|---:|---:|
| DL-02-s42 | LSTM | 81.85% | 1,242,305 | 10 |
| DL-03-s42 | GRU | 81.02% | 1,231,937 | 10 |
| DL-07-s42 | StackedLSTM | 75.56% | 1,275,329 | 10 |
| DL-08-s42 | LSTM_GRU | 81.17% | 1,267,265 | 10 |
| DL-09-s42 | BidirectionalLSTM | 79.58% | 1,284,609 | 10 |
| DL-10-s42 | BidirectionalGRU | 81.77% | 1,263,873 | 10 |

進階組為純架構比較，前一層 return_sequences=True，雙向層使用 Bidirectional；不添加 Dropout 或 EarlyStopping。
參數量不同，不能宣稱公平的等參數量成本比較。

## Notebook、模型、輸出與驗收
- Notebook 04：3 組新 LSTM 條件，重用 DL-02；全部儲存格已執行。
- Notebook 05：4 組進階架構，基準對照 DL-02、DL-03；全部儲存格已執行。
- 14 個模型（4 個 joblib、10 個 Keras），14 份 Validation Prediction CSV。
- 14 張 Confusion Matrix、10 張深度學習 Loss 曲線，加上整體比較圖。
- 機器可讀：results/runs.csv、results/gate_c_acceptance.json；測試結果：results/gate_c_unittest.log。
- 已以 SHA-256 確認 Gate B 既有 7 個模型與相應 Run 原始資料列完全未改動。
- 12 項 Gate C unittest：全部通過（Exit Code 0）。

## 限制與結論
- F1、Accuracy 全部為 Validation 指標，不能作為 Test 泛化結論。
- 此輪僅 seed=42；GPU 部分操作可能非完全決定性。
- 序列截斷 160、Test 內部重複與近似語意重複問題仍需在最終報告揭露。
- **Gate C 已通過**；後續可先完成 Phase 4（07）Validation 比較與模型選擇策略，再進行鎖定後的 Test 評估（Gate E）。
