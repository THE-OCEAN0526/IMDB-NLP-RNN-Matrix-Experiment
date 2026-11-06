# Optional Gate D — GloVe 100 維擴充驗收

**狀態：PASS_POST_GATE_E_VALIDATION_ONLY。**

完成原規格矩陣 E 的兩個選做條件：TL-01 Frozen 與 TL-02 Fine-tuning，並重用 DL-03 原基準。
三組均使用相同 Train／Validation 切分、固定 GRU(64) 架構、Embedding 100 維、10 Epoch、Seed 42。

## Validation 比較（固定第 10 個 Epoch）
| Run | 初始化 | F1 | Accuracy | 可訓練參數 |
|---|---|---:|---:|---:|
| DL-03-s42 | Random Embedding (baseline) | 81.02% | 80.99% | 1,231,937 |
| TL-01-s42 | GloVe Frozen | 79.99% | 81.28% | 31,937 |
| TL-02-s42 | GloVe Fine-tuning | 80.34% | 81.98% | 1,231,937 |

## 驗收及交付
- GloVe 6B／100d 來源 ZIP 及二進位向量檔已驗證，含 400,000 個詞及固定雜湊。
- 11,998 個可比對詞彙中精確匹配 8,637 個（71.99%）；Train 已知 token 加權匹配率 90.95%。
- Notebook 06 完整執行 7/7 程式儲存格、零錯誤；兩個模型各 10 Epoch 完整訓練並儲存。
- Validation 各 4,956 筆預測、Confusion Matrix 和 F1 指標重算完全一致。
- TL-01 Embedding 在訓練後完全保持初始 GloVe 值；TL-02 Embedding 確認已更新。
- 12/12 項選做 Gate D 測試 PASS。
- 所有 Gate A／B／C／D／E 跨階段測試 **57/57 通過**。
- 原本 Gate E 的 26 個關鍵原始檔雜湊一致，沒有重新做 Test。

## 結論及重要限制
- 本輪固定第 10 Epoch 的 Validation F1：DL-03 Random 81.02%、TL-01 Frozen 79.99%、TL-02 Fine-tuning 80.34%。
- TL-02 的 Train/Validation Loss 顯示過度擬合，這是原始固定 10 Epoch 設計的一項觀察，不事後改挑更好 Epoch。
- 資料來源為 Zenodo 上的 GloVe 6B／100d word2vec binary 轉存版（不是 Stanford 原始 txt）；已記錄來源與格式。
- GloVe 選做實驗發生於已鎖定並完成的官方 Gate E 後，因此只能稱為探索性 Validation 比較；不得用先前已看過的 Test 重新選模型。
- 不聲稱跨 Seed 的顯著性或適用於所有 NLP 任務。

詳細訓練歷程、SHA-256、文字向量匹配情況及圖表：optional_glove/results/OPTIONAL_GLOVE_REPORT.md。
