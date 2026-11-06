# IMDB NLP 矩陣實驗 — Phase 4 / Test 前模型比較

**已完成 14 組 Validation 統整與模型選擇候選；此處不是 Gate E，未使用官方 Test。**

## 14 組 Validation 排名（F1 由高至低，平手以 Accuracy / Run ID）
| 排名 | Run | 模型 | Accuracy | Precision | Recall | F1 | Params | Fit 秒 | 推論秒 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ML-04-s42 | TF-IDF + SVC | 89.21% | 88.74% | 89.92% | 89.32% | — | 274.28 | 48.07 |
| 2 | ML-02-s42 | BoW + SVC | 87.67% | 87.71% | 87.75% | 87.73% | — | 194.53 | 34.29 |
| 3 | DL-05-s42 | LSTM + EarlyStopping | 84.54% | 85.03% | 84.01% | 84.52% | 1,242,305 | 9.94 | 0.21 |
| 4 | DL-06-s42 | LSTM + Dropout + ES | 83.66% | 83.46% | 84.13% | 83.79% | 1,242,305 | 12.31 | 0.22 |
| 5 | DL-02-s42 | LSTM | 82.47% | 85.25% | 78.71% | 81.85% | 1,242,305 | 13.67 | 0.17 |
| 6 | DL-10-s42 | Bidirectional GRU | 82.16% | 83.99% | 79.67% | 81.77% | 1,263,873 | 32.69 | 0.31 |
| 7 | DL-08-s42 | LSTM + GRU | 80.67% | 79.45% | 82.97% | 81.17% | 1,267,265 | 33.59 | 0.30 |
| 8 | DL-03-s42 | GRU | 80.99% | 81.25% | 80.80% | 81.02% | 1,231,937 | 12.25 | 0.16 |
| 9 | DL-04-s42 | LSTM + Dropout | 79.62% | 79.08% | 80.80% | 79.93% | 1,242,305 | 24.62 | 0.22 |
| 10 | DL-09-s42 | Bidirectional LSTM | 80.04% | 81.86% | 77.42% | 79.58% | 1,284,609 | 34.43 | 0.34 |
| 11 | ML-03-s42 | TF-IDF + KNN | 76.86% | 75.09% | 80.67% | 77.78% | — | 0.00 | 1.83 |
| 12 | DL-07-s42 | Stacked LSTM | 77.82% | 84.61% | 68.26% | 75.56% | 1,275,329 | 32.70 | 0.32 |
| 13 | DL-01-s42 | SimpleRNN | 70.50% | 69.77% | 72.80% | 71.25% | 1,210,625 | 18.84 | 0.43 |
| 14 | ML-01-s42 | BoW + KNN | 64.35% | 63.68% | 67.50% | 65.54% | — | 0.01 | 2.08 |

## Test 前候選（不根據 Test 調整）
- 全組最高：**ML-04-s42（TF-IDF + SVC）**。
- 傳統機器學習代表：**ML-04-s42（TF-IDF + SVC）**。
- 深度學習代表：**DL-05-s42（LSTM + EarlyStopping）**。
- 進入下一階段前，將以 model_selection_lock.json 鎖定兩個類別代表；建議官方 Test 只評估這兩組各一次。

## 研究問題結論（Validation F1 百分點差；單次 seed=42）

### RQ1
- TFIDF_vs_BoW_given_KNN: +12.25 百分點
- TFIDF_vs_BoW_given_SVC: +1.59 百分點

### RQ2
- SVC_vs_KNN_given_BoW: +22.19 百分點
- SVC_vs_KNN_given_TFIDF: +11.54 百分點

### RQ3
- LSTM_vs_SimpleRNN: +10.59 百分點
- GRU_vs_SimpleRNN: +9.77 百分點
- LSTM_vs_GRU: +0.82 百分點

### RQ4
- Dropout_only_vs_baseline: -1.92 百分點
- ES_only_vs_baseline: +2.67 百分點
- Both_vs_baseline: +1.95 百分點
- interaction_difference_in_differences: +1.19 百分點

### RQ5
- StackedLSTM_vs_LSTM: -6.29 百分點
- LSTM_GRU_vs_LSTM: -0.68 百分點
- BiLSTM_vs_LSTM: -2.27 百分點
- BiGRU_vs_GRU: +0.75 百分點

## 圖表
- results/figures/07_all_14_validation_f1.png
- results/figures/07_traditional_2x2_f1.png
- results/figures/07_regularization_2x2_f1.png
- results/figures/07_advanced_rnn_f1.png
- results/figures/07_deep_learning_cost_f1.png
- 既有混淆矩陣：每模型 1 張；既有 Loss curves：每個 DL 模型 1 張。Notebook 07 直接引用，不改舊結果。

## 評估限制
- 全部都是 Validation；不能當作獨立 Test 泛化成績。
- 單一 seed 42，差值未進行顯著性檢定。早停組實際 Epoch 不同。
- 模型參數量並非匹配；GPU/CPU 訓練時間計算範圍也不相同。
- KNN fit 時間不包含昂貴的距離查詢；SVC fit 秒數含兩個候選 C 的訓練；DL 含每個 Epoch 的 Validation。
- 160-token 輸入截斷超過半數評論；官方 Test 內部重複保留；未稽核語意近似重複。
- 選做 GloVe（TL-01／TL-02）未訓練，不列入 14 個必做條件。

## Gate E 尚待完成
先凍結 selection lock，再進行事先決定範圍的單次官方 Test 評估；不能看 Test 再改模型、選參數或改變正式候選。
