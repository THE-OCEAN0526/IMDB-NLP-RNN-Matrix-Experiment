"""Generate v1.1 isolated Phase 0 notebooks (never overwrite any v1.0 notebooks)."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/"notebooks"
TARGET.mkdir(parents=True,exist_ok=True)
setup="""from pathlib import Path
import json
import sys
ROOT=Path('/workspace/imdb_matrix_experiment/variants/dedup_v1_1')
assert ROOT.exists()
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
"""
def md(s):return nbf.v4.new_markdown_cell(s)
def py(s):return nbf.v4.new_code_cell(s)
def save(name,cells):
    file=TARGET/name
    assert not file.exists(),f"Refuse to overwrite {file}"
    notebook=nbf.v4.new_notebook(cells=cells,metadata={
        "kernelspec":{"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"},
        "language_info":{"name":"python"},
    })
    nbf.write(notebook,file)
    print("CREATED",file)

save("00_IMDB_Data_Preparation.ipynb",[
    md("# 00｜IMDB v1.1 去重版資料準備\n\n**依批准的 v1.1 規格執行。** 不覆蓋舊 v1.0。"
       "完整清理後文字 SHA-256 作為重複判定；官方 Test 25,000 筆完全保留（包含其內部重複）。"
       "\n\n資料流程：來源驗證 → 官方 Test 保留 → 排除 Train 中與 Test 相同文本 123 筆"
       " → 訓練池內去重 98 筆 → seed 42 分層 80/20 切分成 19,823 / 4,956。"),
    py(setup+"""from src.data_utils import prepare
from src.common import config,file_hash,load_splits
import pandas as pd
config_obj, config_sha=config()
print("Variant:",config_obj["version"])
print("Config SHA256:",config_sha)
for name in ("train","test"):
    source=Path(config_obj["dataset"][f"source_{name}"])
    actual=file_hash(source)
    assert actual==config_obj["dataset"][f"sha256_{name}"]
    print("Source:",name,source.stat().st_size,actual)
"""),
    md("## 1. 重建獨立 v1.1 切分與稽核清單\n"
       "系統會拒絕已變更的來源 SHA 或與批准數量及 ID 雜湊不同的資料切分。"
       " Train/Validation/官方 Test 的原始資料與清理後完整文字由專案內 manifest 追蹤。"),
    py("""manifest=prepare()
print("Counts:")
print(pd.DataFrame(manifest["counts"]).T.to_string())
print("Exclusions:",manifest["audit"]["excluded_test_overlap"],manifest["audit"]["excluded_intra_train_duplicates"])
print("Split SHA:",manifest["split_assignments_sha256"])
"""),
    md("## 2. 防洩漏檢查：不同 split 的完整清理後文本沒有完全重複\n"
       "檢查的是完整清理後字串，不是 160-token 截斷序列。"
       " 並非語意相似度／近似重複檢查。"),
    py("""splits=load_splits()
hashes={n:set(df.cleaned_hash) for n,df in splits.items()}
assert not(hashes["train"]&hashes["validation"])
assert not(hashes["train"]&hashes["test"])
assert not(hashes["validation"]&hashes["test"])
assert len(splits["train"])==19823 and len(splits["validation"])==4956 and len(splits["test"])==25000
print("Cross-split identical cleaned text: 0 / 0 / 0")
for name,df in splits.items():
    print(name,len(df),df.label.value_counts().sort_index().to_dict())
print("Official Test retained:",manifest["audit"]["official_test_rows"])
"""),
    md("## 3. 測試集限制與排除明細\n"
       "官方 Test 自身的 201 筆額外重複仍予保留，避免破壞官方 Test 25k 基準。"
       " 排除資料僅儲存 ID、hash、label 與原因，不散佈評論全文。"),
    py("""audit=manifest["audit"]
print("Test internal duplicate groups:",audit["test_internal_duplicate_groups"])
print("Test extra duplicate rows:",audit["test_internal_extra_duplicate_rows"])
print("Caveat:",audit["warning"])
for name in ("excluded_test_overlap.csv","excluded_intra_train_duplicates.csv"):
    df=pd.read_csv(ROOT/"data/processed"/name)
    print(name,"rows",len(df),"labels",df.label.value_counts().sort_index().to_dict())
"""),
])
save("01_IMDB_Preprocessing_Vectorization.ipynb",[
    md("# 01｜IMDB v1.1 去重版文字清理與向量化\n\n"
       "Train/Validation/Test 使用同一份已核定的清理規則和切分檔案。"
       " BoW、TF-IDF、Keras TextVectorization **全部重新從 v1.1 Train fit/adapt**，"
       "絕不重用 v1.0 的詞彙表或 Token IDs。"
       "\n\n保留否定詞及標點，TextVectorization 使用 12,000 詞彙、長度 160。"),
    py(setup+"""from src.common import config,load_splits,clean_review
from src.feature_utils import prepare_features
import json
import pandas as pd
import numpy as np
c,c_hash=config()
manifest=json.loads((ROOT/"data/processed/split_manifest.json").read_text())
assert manifest["config_sha256"]==c_hash
print("Variant",c["version"],"preprocessing",c["preprocessing"]["version"])
print("Canonical split SHA",manifest["split_assignments_sha256"])
"""),
    md("## 1. 相同清理函式對照\n"
       "HTML entity → <br> 空白 → 其他 HTML 標籤去除 → 小寫 → 空白規範化；標點和 stopwords 保留。"),
    py("""df=load_splits()["train"]
for i in range(2):
    original=df.iloc[i].text
    cleaned=clean_review(original)
    assert cleaned==df.iloc[i].cleaned_text
    print("RAW:",original[:190].replace("\\n"," "))
    print("CLEAN:",cleaned[:190])
    print("---")
"""),
    md("## 2. v1.1 Train-only 擬合與 Validation/Test 轉換\n"
       "只用新的 19,823 筆 Train 學習詞彙表及 IDF；Validation/Test 僅 transform/inference。"
       "另保存六個稀疏矩陣供 Phase 1 使用。"),
    py("""features=prepare_features()
print("Fitted on:",features["fitted_on"])
print("Feature dimensions:",features["features"])
print("Split SHA:",features["manifest_split_sha256"])
print("Fit policy:",features["fit_policy"])
"""),
    md("## 3. Keras Token ID 和處理結果檢查\n"
       "ID=0 代表 Padding，ID=1 代表未知詞 OOV；所有輸入固定 160 tokens。"),
    py("""art=ROOT/"artifacts/vectorizers"
vocab=(art/"keras_vocabulary_12000.txt").read_text().splitlines()
assert vocab[0]=="" and vocab[1]=="[UNK]"
with np.load(art/"keras_token_ids_12000_160.npz",allow_pickle=False) as x:
    for name in ("train","validation","test"):
        print(name,x[f"X_{name}"].shape,x[f"y_{name}"].shape)
    print("First 20 token IDs:",x["X_train"][0,:20].tolist())
print("Vocab first 15:",vocab[:15])
"""),
    md("## 4. 長評論截斷／OOV 分析與安全限制\n"
       "原規格維持 max_len=160；在正式報告列出截斷比例。"
       " Gate A 專注於完全相同清理後全文；近似語意仍是限制。"),
    py("""print(pd.DataFrame(features["lengths"]).T.to_string())
print("Feature metadata:", ROOT/"results/feature_metadata.json")
print("Note: Deduplicated v1.1 features cannot be mixed with v1.0 models.")
"""),
])
