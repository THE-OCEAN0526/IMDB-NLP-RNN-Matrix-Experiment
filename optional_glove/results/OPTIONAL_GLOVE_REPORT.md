# IMDB 矩陣實驗 — GloVe 100d 選做實作（補做）

**狀態：兩組已實際訓練並保存；此為 Gate E 完成後的探索性 Validation 擴充，不重新使用官方 Test。**

## 1. 研究問題與實驗設計
- RQ6：GloVe 是否能改善 GRU 的分類表現？比較原 DL-03 (Random, trainable) 與新增 TL-01 (GloVe Frozen)、TL-02 (GloVe Fine-tuning)。
- 固定 dedup_v1.1；Train 19,823／Validation 4,956；原始 Train-only Keras TextVectorization，12,000 vocabulary；長度 160。
- Embedding 100 維、GRU 64 units、sigmoid、Adam(0.001)、BCE、batch 128、seed=42、10 Epoch、無 Dropout／EarlyStopping。
- TL-01 只有 31,937 個參數可訓練；TL-02 以及原基準 DL-03 均有 1,231,937 個參數可訓練。
- 沒有重新訓練 DL-03，沒有改變原始 14 組結果或已鎖定的 Gate E 測試。

## 2. GloVe 來源與詞彙對齊
- Stanford GloVe 6B：Wikipedia 2014 + Gigaword 5，400,000 詞，100 維、uncased。
- 來源網站：https://nlp.stanford.edu/projects/glove/
- 本次下載鏡像：https://zenodo.org/records/4925376/files/glove.6B.100d.zip?download=1
- **格式註記：Zenodo 供應的是將 GloVe 向量轉存的 word2vec float32 binary，並非 Stanford 原始純文字檔。**
- 100d 壓縮包 ZIP MD5：f19871e3053750198004fb2acc1f8d44
- 二進位向量 SHA-256：4e29827d144c2097654433b1c4be41ce320949f0792ab820991a68fa3e34d494
- 對齊矩陣 SHA-256：0ddb2fdf9c12ca310494649da8225bbfe74b650104fa31bd90859e44a8862aee
- 授權：Open Data Commons Public Domain Dedication and License 1.0
- Train-only vocabulary 的 11,998 個可匹配詞中，精確匹配 8,637 個（71.99%）。
- 依 Train 已知 token 次數加權，GloVe 精確匹配覆蓋率 90.95%。
- PAD=0 對應零向量並啟用 mask_zero=True；UNK=1 及未命中詞使用 seed=42 的 N(0,0.05) 初始化。
- Validation／Test 均未參與詞彙表 fit 或向量對齊；拼字或標點附著使部分詞無法與 GloVe 精確對應。

## 3. 三種 Embedding × GRU 的 Validation 實際成績
| Run | Embedding | Trainable Params | Epoch | Val Accuracy | Val Precision | Val Recall | Val F1 | Val Loss | Fit 秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DL-03-s42 | Random Trainable | 1,231,937 | 10 | 80.99% | 81.25% | 80.80% | 81.02% | 0.7745 | 12.25 |
| TL-01-s42 | GloVe_100d_Frozen | 31,937 | 10 | 81.28% | 86.32% | 74.53% | 79.99% | 0.4084 | 23.00 |
| TL-02-s42 | GloVe_100d_FineTuning | 1,231,937 | 10 | 81.98% | 88.85% | 73.32% | 80.34% | 0.9013 | 23.17 |

- TL-01 Frozen 相對 DL-03 Random：-1.03 Validation F1 百分點。
- TL-02 Fine-tuning 相對 DL-03 Random：-0.68 Validation F1 百分點。
- TL-02 Fine-tuning 相對 TL-01 Frozen：+0.35 Validation F1 百分點。

## 4. 訓練 Loss 與過度擬合
- TL-01：最佳 Validation Loss 出現在第 9 個 Epoch；最終 Val Loss 0.4084。
- TL-02：最佳 Validation Loss 出現在第 2 個 Epoch；最終 Val Loss 0.9013。
- TL-02 第 10 Epoch Train Accuracy 99.53% ，Validation Accuracy 81.98% ，可見過度擬合跡象。
- 本組為架構控制實驗，因此刻意不使用 EarlyStopping；沒有擷取較佳早期權重作為另外一個未宣告模型。

## 5. 產物與可追溯性
- Notebook：notebooks/06_IMDB_GloVe_Transfer_Learning.ipynb。
- Source：optional_glove/source/glove.6B.100d.zip、optional_glove/source/glove.6B.100d。
- 對齊矩陣、配對 mask：optional_glove/artifacts/。
- TL-01 與 TL-02 的 Keras 訓練模型、10 Epoch history、Validation 逐筆預測與 Confusion Matrix 分別保留。
- 來源、參數及 GPU 執行日誌：optional_glove/results/ 和 optional_glove/preparation.log。
- 三組比較表：optional_glove/results/three_gru_validation_comparison.csv。
- 圖表：optional_glove/results/glove_3way_validation_f1.png、glove_3way_validation_accuracy.png。

## 6. 研究結論與限制
- 在本次固定 10 Epoch、單次 seed=42 的 Validation 條件下，使用 GloVe 並非必然優於原本隨機初始化 Embedding 的 GRU。
- GloVe 是外部預訓練詞向量，不代表 GRU 分類器已被預訓練；Fine-tuning 和 Frozen 也有不同可訓練參數量。
- 此選做實驗為原 Gate E **完成後**的探索性研究，不能把先前官方 Test 的分數用來重新選擇最優模型。
- 沒有對 TL-01 / TL-02 進行任何官方 Test 推論，也沒有更改原先 Test 前鎖定的 ML-04／DL-05 或 Test 25,000 筆評估紀錄。
- 此輪沒有多種子統計檢定，也不應宣稱差異有統計顯著性。
