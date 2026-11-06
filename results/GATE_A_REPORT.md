# IMDB NLP 矩陣實驗 — v1.1 去重版 Phase 0 / Gate A 驗收報告

驗收狀態：**PASS（已確認 Train／Validation／Test 間完全相同的清理後完整文字為零）**

## 1. 檔案位置與版本
- 執行環境：ai-server / Docker cnn-jupyterlab / 自有 JupyterLab
- 專案獨立變體：/home/xuan/cnn-workspace/imdb_matrix_experiment/variants/dedup_v1_1/
- v1.0 所有來源、資料、Notebook、設定、結果保持不變（SHA-256 已比對）
- v1.1 設定 SHA-256：6483ed250a1ddda6e54da5f6120e41180c8b0215bc90cf4ac68f71c874b64664
- 切分清單 SHA-256：9e25b0d714d1fb555c8c8a58fa6858c392ec60ecfba554b923553626154174e9
- 原始 Train/Test SHA-256 已核對，官方 Test 25,000 筆及順序完全保持不變

## 2. 資料切分與篩選
| Split | 筆數 | 負面 0 | 正面 1 | 唯一清理後全文數 |
|---|---:|---:|---:|---:|
| train | 19823 | 9869 | 9954 | 19823 |
| validation | 4956 | 2467 | 2489 | 4956 |
| test | 25000 | 12500 | 12500 | 24799 |

- 原 Train 中與官方 Test 相同文字排除：123 筆（負面 95／正面 28）
- 餘下 Train 池內去重：98 筆（負面 69／正面 29）
- 兩份排除清單僅記錄樣本 ID、索引、標籤、Hash 和理由；沒有散佈原文
- 官方 Test 內部重複群組：193，額外重複資料列：201（刻意保留）

## 3. Gate A 交叉重複與資料洩漏規範
| 原文 hash 比對（完整清理後文字） | 重複組數 |
|---|---:|
| Train / Validation | 0 |
| Train / Test | 0 |
| Validation / Test | 0 |

- 清理版本：phase0-text-v1（與 v1.0 相同）
- 同一全文 hash 但異標籤群組：0
- 所有 BoW、TF-IDF、Keras TextVectorization fit / adapt 均僅使用新 v1.1 Train；其他 splits 只 transform / inference
- 索引以 source_index 排序，seed=42、stratify=label；輸出 ID 雜湊比對批准的模擬值
- 本結果僅表示已消除完全相同清理後全文的跨組重複，不涵蓋語意近似重複

## 4. Phase 0 Notebook 和向量化成果
| Notebook | 代碼儲存格 | 執行成功 | 儲存錯誤 |
|---|---:|---:|---:|
| 00_IMDB_Data_Preparation.ipynb | 4 | 4 | 0 |
| 01_IMDB_Preprocessing_Vectorization.ipynb | 5 | 5 | 0 |

- BoW 特徵：12,000；TF-IDF 特徵：12,000；Keras 詞彙表：12,000
- Keras 序列長度：160，並儲存 3 個 split 的 token IDs、Labels、Sample IDs
- 6 個 SciPy sparse 特徵矩陣與 2 個已擬合 scikit-learn 向量器皆已保存

### Token 長度統計
| Split | 超出 160 tokens 的評論 | 比例 |
|---|---:|---:|
| train | 10941 | 55.19% |
| validation | 2743 | 55.35% |
| test | 13583 | 54.33% |

## 5. 自動驗收
- Gate A 11/11 項 unittest：**全部通過，Exit Code 0**
- 驗收日誌：results/gate_a_unittest.log
- 機器可讀報告：results/gate_a_acceptance.json
- 已比對 v1.0 兩本 Notebook、原設定及 manifest SHA-256，均未變更
- 曾修正一項測試：TF-IDF CSR 的元素順序不同，但依矩陣索引比較最大差約 6.66e-16，屬浮點誤差；沒有修改向量化結果

## 6. 仍須揭露的限制
1. 官方 Test 內部原有重複文本仍保留；其測試成績不是去重後的唯一文本測試指標。
2. 未做語意相同／近似重複評論檢查，因此不能宣稱排除所有可能的資料洩漏。
3. 序列長度 160 導致約 54–55% 評論被截斷；後續若調整長度，必須明確建立新實驗條件。
4. 尚未開始 Phase 1 模型訓練，也沒有使用 Test 成績調參。

## 7. 放行結論
v1.1 的 Phase 0 實作與 Gate A（限定完全相同清理後全文的跨組重複）已通過，可供後續 Phase 1 新模型矩陣使用；所有模型必須使用此 v1.1 切分與詞彙表重新訓練，不得混用 v1.0 模型成績。
