"""Generate teaching-quality 02/03 notebooks without overwriting existing files."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS=ROOT/"notebooks"
NOTEBOOKS.mkdir(parents=True,exist_ok=True)

def mk(name,cells):
    path=NOTEBOOKS/name
    assert not path.exists(),"Refuse existing file: "+str(path)
    book=nb.v4.new_notebook(cells=cells,metadata={
        "kernelspec":{"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"},
        "language_info":{"name":"python"},
    })
    nb.write(book,path)
    print("CREATED",path)

def md(text):return nb.v4.new_markdown_cell(text)
def code(text):return nb.v4.new_code_cell(text)

init="""from pathlib import Path
import json,sys
ROOT=Path('/workspace/imdb_matrix_experiment/variants/dedup_v1_1')
assert ROOT.is_dir()
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from IPython.display import display,Image
from src.evaluation import phase1_settings
settings,phase1_hash,manifest=phase1_settings()
print('Variant:',manifest['version'])
print('Split SHA256:',manifest['split_assignments_sha256'])
print('Phase1 SHA256:',phase1_hash)
assert settings['test_policy']=='do_not_evaluate_in_phase1'
"""

ml=[
 md("""# 02｜傳統機器學習：2×2 基準矩陣
| Run | Representation | Classifier |
|---|---|---|
| ML-01 | BoW | KNN（cosine brute） |
| ML-02 | BoW | SVC（linear kernel） |
| ML-03 | TF-IDF | KNN（cosine brute） |
| ML-04 | TF-IDF | SVC（linear kernel） |

本 Notebook 只使用 v1.1 Train 19,823、Validation 4,956。Validation F1 用來選 k / C，官方 Test 保留到 Gate E。
"""),
 code(init+"""from src.evaluation import subset_data
train,val=subset_data()
assert len(train)==19823 and len(val)==4956
print('Train',len(train),'Validation',len(val))
print('KNN candidates:',settings['ml']['knn']['k_candidates'])
print('SVC candidates:',settings['ml']['svc']['C_candidates'])
"""),
 md("## 1. 共用特徵矩陣\n向量器於 Phase 0 v1.1 僅 fit Train。KNN / SVC 使用相同 sparse matrix。"),
 code("""from scipy.sparse import load_npz
art=ROOT/'artifacts/vectorizers'
for name in ('bow','tfidf'):
    X=load_npz(art/(name+'_train.npz'))
    V=load_npz(art/(name+'_validation.npz'))
    assert X.shape==(19823,12000) and V.shape==(4956,12000)
    print(name,'train',X.shape,'validation',V.shape,
          'train sparsity:',round(1-X.nnz/(X.shape[0]*X.shape[1]),4))
"""),
 md("""## 2. 正式訓練四個基準條件
原生 sklearn KNeighborsClassifier：metric=cosine、algorithm=brute、k ∈ {3,5}。
原生 sklearn.svm.SVC：kernel=linear、C ∈ {0.1,1.0}。相同候選範圍，各組只用 Validation F1 定案。

已存在完整 Run 的模型與預測不重訓；僅補做缺漏組別。
每組完整訓練、候選指標、預測耗時及選擇理由皆保留。
"""),
 code("""import subprocess
wanted=['ML-01-s42','ML-02-s42','ML-03-s42','ML-04-s42']
registry=ROOT/'results/runs.csv'
existing=set(pd.read_csv(registry).run_id) if registry.exists() else set()
missing=[name[:-4] for name in wanted if name not in existing]
print('Missing full-data models:',missing)
if missing:
    subprocess.run([sys.executable,'-B','-m','src.phase1_ml','--runs',*missing],cwd=ROOT,check=True)
print('Four baseline conditions completed.')
"""),
 md("## 3. Validation 指標與超參數紀錄\n各模型同時計算 Accuracy、Precision、Recall、F1、Confusion Matrix；測試集尚未參與。"),
 code("""rows=pd.read_csv(ROOT/'results/runs.csv')
selected=rows[rows.run_id.isin(wanted)].sort_values('run_id')
assert len(selected)==4 and selected.run_id.is_unique
display(selected[['run_id','representation','model_name','val_accuracy','val_precision',
                  'val_recall','val_f1','train_time_sec','inference_time_sec']].round(4))
for rid in selected.run_id:
    diag=json.loads((ROOT/('results/'+rid+'_candidate_selection.json')).read_text())
    print(rid,'selected:',diag['chosen_parameter'],
          'validation candidates:',[(x['value'],round(x['val_f1'],4)) for x in diag['candidates']])
"""),
 md("## 4. 圖表：Validation F1 與混淆矩陣\n所有圖表均來自 Validation，未使用 Test。"),
 code("""import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,3.8))
ax.bar(selected.run_id,selected.val_f1)
ax.set(ylim=(0,1),ylabel='Validation F1',title='ML Baselines — dedup_v1.1')
ax.grid(axis='y',alpha=.2)
plt.show()
for rid in selected.run_id:
    print(rid)
    display(Image(filename=str(ROOT/('results/figures/'+rid+'_val_confusion_matrix.png')),width=360))
"""),
 md("## Phase 1 ML 結果\n正式模型保存於 artifacts/models，候選紀錄與 predictions 保存於 results。\nKNN 的高維余弦暴力搜尋耗時需與 SVC 的訓練時間分開解釋。")
]
dl=[
 md("""# 03｜深度學習：單層 RNN 家族
| Run | Architecture |
|---|---|
| DL-01 | Embedding(100) → SimpleRNN(64) → Dense(1,sigmoid) |
| DL-02 | Embedding(100) → LSTM(64) → Dense(1,sigmoid) |
| DL-03 | Embedding(100) → GRU(64) → Dense(1,sigmoid) |

共用 12,000 詞彙、160 tokens、Adam(0.001)、BCE、batch=128、最多 10 epochs，無 Dropout 與 Early Stopping。固定 seed=42；只使用 Validation，不接觸官方 Test。
"""),
 code(init+"""import tensorflow as tf
print('TensorFlow',tf.__version__,'GPU',tf.config.list_physical_devices('GPU'))
assert tf.config.list_physical_devices('GPU')
print('Training configuration',settings['deep_learning'])
"""),
 md("## 1. 共用 Token IDs 與相同資料分割\n資料由 Phase 0 v1.1 的 Train-only TextVectorization 產生。"),
 code("""with np.load(ROOT/'artifacts/vectorizers/keras_token_ids_12000_160.npz',allow_pickle=False) as data:
    for split,rows in (('train',19823),('validation',4956)):
        assert data['X_'+split].shape==(rows,160)
        print(split,data['X_'+split].shape,data['y_'+split].shape)
"""),
 md("## 2. 模型工廠與參數量\n三組只改變循環層；hidden units 相等，總參數量不必然相等。"),
 code("""from src.model_factory import make_single_layer_rnn,BASELINES
for run_id in BASELINES:
    tf.keras.utils.set_random_seed(42)
    model=make_single_layer_rnn(run_id)
    print(run_id,BASELINES[run_id],model.count_params(),
          [x.__class__.__name__ for x in model.layers])
    del model
    tf.keras.backend.clear_session()
"""),
 md("""## 3. GPU 上實際訓練 DL-01、DL-02、DL-03
三組相同預設訓練條件：10 epochs、無 dropout、無 early stopping。
正式執行的模型、loss 曲線、Validation 預測結果會保存到獨立 v1.1 專案。
已有完成且保存的 Run 則不重訓。
"""),
 code("""import subprocess
wanted=['DL-01-s42','DL-02-s42','DL-03-s42']
registry=ROOT/'results/runs.csv'
existing=set(pd.read_csv(registry).run_id) if registry.exists() else set()
missing=[name[:-4] for name in wanted if name not in existing]
print('Missing full-data models:',missing)
if missing:
    subprocess.run([sys.executable,'-B','-m','src.phase1_rnn','--runs',*missing],cwd=ROOT,check=True)
print('Three RNN family baselines completed.')
"""),
 md("## 4. Validation 分數及訓練成本\nAccuracy、Precision、Recall、F1、最後 epoch 的 Validation Loss、參數量、訓練時間及推論時間。"),
 code("""rows=pd.read_csv(ROOT/'results/runs.csv')
selected=rows[rows.run_id.isin(wanted)].sort_values('run_id')
assert len(selected)==3 and selected.run_id.is_unique
display(selected[['run_id','model_name','val_accuracy','val_precision','val_recall','val_f1',
                  'val_loss','trainable_params','epochs_completed',
                  'train_time_sec','inference_time_sec']].round(4))
"""),
 md("## 5. Loss 曲線與混淆矩陣\n10 epoch 曲線允許觀察過度擬合；此基準不使用 EarlyStopping。"),
 code("""import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,3.8))
ax.bar(selected.run_id,selected.val_f1)
ax.set(ylim=(0,1),ylabel='Validation F1',title='Single-layer RNN — dedup_v1.1')
ax.grid(axis='y',alpha=.2)
plt.show()
for rid in selected.run_id:
    print(rid)
    display(Image(filename=str(ROOT/('results/figures/'+rid+'_learning_curve.png')),width=540))
    display(Image(filename=str(ROOT/('results/figures/'+rid+'_val_confusion_matrix.png')),width=360))
"""),
 md("## Phase 1 DL 結果\nDL-02 是 Phase 2 LSTM Dropout×EarlyStopping 比較中不重訓的基準。\n單次 seed 的高低不是統計顯著性結論；後續依 Validation 而非 Test 選模型。")
]
mk("02_IMDB_Traditional_ML_Matrix.ipynb",ml)
mk("03_IMDB_RNN_Baselines.ipynb",dl)
