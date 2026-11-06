"""Create two Phase 2 executable teaching notebooks; never touch 00..03."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
NB=ROOT/"notebooks"
def M(text):return nbf.v4.new_markdown_cell(text)
def C(text):return nbf.v4.new_code_cell(text)
def save(name,cells):
    dest=NB/name
    assert not dest.exists(),f"Refuse to overwrite existing notebook {dest}"
    nb=nbf.v4.new_notebook(cells=cells,metadata={
       "kernelspec":{"display_name":"Python 3 (ipykernel)","name":"python3","language":"python"},
       "language_info":{"name":"python"},
    })
    nbf.write(nb,dest)
    print("CREATED",dest)

intro="""from pathlib import Path
import json,sys
ROOT=Path('/workspace/imdb_matrix_experiment/variants/dedup_v1_1')
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import yaml,numpy as np,pandas as pd
from IPython.display import display,Image
from src.common import file_hash
from src.evaluation import phase1_settings
base,base_sha,manifest=phase1_settings()
p=ROOT/'config/phase2_conditions.yaml'
cfg=yaml.safe_load(p.read_text(encoding='utf8'))
assert cfg['phase1_config_sha256']==base_sha
assert cfg['split_assignments_sha256']==manifest['split_assignments_sha256']
assert cfg['test_policy']=='do_not_evaluate_until_gate_e'
print('Variant:',manifest['version'],'Gate A SHA:',manifest['split_assignments_sha256'])
print('Phase 2 SHA:',file_hash(p))
print('Test policy:',cfg['test_policy'])
"""

regular=[
 M("""# 04｜LSTM：Dropout × Early Stopping 2×2 對照

固定架構：Embedding(12,000,100) → LSTM(64) → 可選 Dropout(0.3) → Dense(1,sigmoid)。

| Run | Dropout | Early Stopping | 是否重新訓練 |
|---|---|---|---|
| DL-02（Gate B 基準） | 無 | 無 | 不重訓 |
| DL-04 | 有，0.3 | 無 | 新增 |
| DL-05 | 無 | 有，patience=3 | 新增 |
| DL-06 | 有，0.3 | 有，patience=3 | 新增 |

EarlyStopping 僅監看 Validation 的 val_loss，restore_best_weights=True。Epoch 上限 10；使用同一份 v1.1 Train / Validation；官方 Test 不參與。
"""),
 C(intro+"""from src.evaluation import subset_data
train,val=subset_data()
assert len(train)==19823 and len(val)==4956
print('Shared:',cfg['shared'])
print('Regularization:',cfg['regularization'])
"""),
 M("## 1. 建模邏輯與控制變因\n三個新增條件都只有單層 LSTM；Early Stopping 是訓練 callback，Dropout 是訓練時的隨機正則化層，而非另一種 RNN 架構。"),
 C("""import tensorflow as tf
from src.phase2_factory import build_phase2
assert tf.config.list_physical_devices('GPU')
for run in ('DL-04','DL-05','DL-06'):
    tf.keras.utils.set_random_seed(42)
    model=build_phase2(run,cfg['shared'],cfg['regularization']['dropout'])
    print(run,'params:',model.count_params(),'layers:',
          [layer.__class__.__name__ for layer in model.layers])
    del model
    tf.keras.backend.clear_session()
"""),
 M("## 2. 執行三個新增條件\n完整記錄訓練 epoch、最佳 val_loss epoch 與還原權重後的 Validation 評估結果。若已有完成 Run，僅載入，不覆蓋 Gate B 的 DL-02。"),
 C("""import subprocess
needed=['DL-04-s42','DL-05-s42','DL-06-s42']
runs=pd.read_csv(ROOT/'results/runs.csv')
assert (runs.run_id=='DL-02-s42').sum()==1,'Missing Phase 1 LSTM baseline'
missing=[x[:-4] for x in needed if x not in set(runs.run_id)]
print('New runs needed:',missing)
if missing:
    subprocess.run([sys.executable,'-B','-m','src.phase2_train','--runs',*missing],cwd=ROOT,check=True)
print('Complete.')
"""),
 M("## 3. 2×2 Validation 對照表\n這個矩陣比較正則化與提早停止的個別及交互作用。EarlyStopping 的正式分數取還原權重後模型，而不是最後訓練 epoch 的歷史列。"),
 C("""ids=['DL-02-s42']+needed
rows=pd.read_csv(ROOT/'results/runs.csv')
sel=rows[rows.run_id.isin(ids)].set_index('run_id').loc[ids].reset_index()
assert len(sel)==4
output=sel[['run_id','val_accuracy','val_precision','val_recall','val_f1','val_loss',
            'epochs_completed','trainable_params','train_time_sec']]
display(output.round(4))
for rid in needed:
    md=json.loads((ROOT/f'results/{rid}_training_metadata.json').read_text())
    print(rid,'condition=',md['condition'],
          'trained epochs=',md['epochs_completed'],
          'best val_loss epoch=',md['best_val_loss_epoch'],
          'restored weights=',md['weights_restored'],
          'final val_loss=',round(md['final_model_val_loss'],5))
"""),
 M("## 4. 學習曲線與混淆矩陣\n觀察 Validation Loss 在無早停時是否上升。EarlyStopping 可能只訓練少於 10 epochs；不能隱藏實際 Epoch 差異。"),
 C("""import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,3.4))
ax.bar(sel.run_id,sel.val_f1)
ax.set(ylim=(0,1),ylabel='Validation F1',title='LSTM Dropout x EarlyStopping (dedup_v1.1)')
ax.grid(axis='y',alpha=.2)
plt.show()
for rid in ids:
    display(Image(filename=str(ROOT/f'results/figures/{rid}_learning_curve.png'),width=490))
    display(Image(filename=str(ROOT/f'results/figures/{rid}_val_confusion_matrix.png'),width=330))
"""),
 M("## 小結\n所有 2×2 分數都是 Validation 成績，單次 seed=42；此階段不得用 Test 反覆調整正則化設定。DL-02 直接沿用 Gate B 原模型，不重複訓練。")
]
advanced=[
 M("""# 05｜進階 RNN 架構：Stacked、Hybrid、Bidirectional

| Run | 架構（Embedding 後） | 對照 |
|---|---|---|
| DL-07 | LSTM(64, return_sequences=True) → LSTM(64) | DL-02 |
| DL-08 | LSTM(64, return_sequences=True) → GRU(64) | DL-02／DL-03 |
| DL-09 | Bidirectional(LSTM(64)) | DL-02 |
| DL-10 | Bidirectional(GRU(64)) | DL-03 |

不加入 Dropout 或 EarlyStopping，保留共同 10 Epoch、Adam(0.001)、batch=128、同一 v1.1 Token IDs。這是架構比較而非等參數量比較；記錄實際參數量與耗時。
"""),
 C(intro+"""from src.evaluation import subset_data
train,val=subset_data()
assert len(train)==19823 and len(val)==4956
assert cfg['shared']['epochs']==10
print('Architectures:',cfg['architectures'])
"""),
 M("## 1. 設計關鍵：輸出序列與雙向連結\n兩層 LSTM / LSTM+GRU 的第一層必須 return_sequences=True，否則下一層收不到完整時間序列。雙向網路將正向和逆向狀態組合。"),
 C("""import tensorflow as tf
from src.phase2_factory import build_phase2
assert tf.config.list_physical_devices('GPU')
for rid in ('DL-07','DL-08','DL-09','DL-10'):
    tf.keras.utils.set_random_seed(42)
    m=build_phase2(rid,cfg['shared'])
    print(rid,cfg['architectures'][rid],m.count_params(),
          [(layer.name,layer.__class__.__name__) for layer in m.layers])
    if rid in ('DL-07','DL-08'):
        assert m.get_layer('lstm_1').return_sequences
    del m
    tf.keras.backend.clear_session()
"""),
 M("## 2. 執行四個進階架構\n模型全部訓練 10 epochs，固定相同驗證程序，保存模型、loss history、Validation 預測與混淆矩陣。"),
 C("""import subprocess
wanted=[f'DL-{i:02d}-s42' for i in range(7,11)]
rows=pd.read_csv(ROOT/'results/runs.csv')
missing=[rid[:-4] for rid in wanted if rid not in set(rows.run_id)]
print('Missing advanced runs:',missing)
if missing:
    subprocess.run([sys.executable,'-B','-m','src.phase2_train','--runs',*missing],cwd=ROOT,check=True)
print('Four advanced architectures completed.')
"""),
 M("## 3. Validation 效能、參數量與成本\n對比單層 LSTM、GRU 基準（DL-02、DL-03），不要把不同參數量的模型誤認為完全等成本比較。"),
 C("""baseline=['DL-02-s42','DL-03-s42']
ids=baseline+wanted
records=pd.read_csv(ROOT/'results/runs.csv')
selected=records[records.run_id.isin(ids)].set_index('run_id').loc[ids].reset_index()
assert len(selected)==6
display(selected[['run_id','model_name','val_accuracy','val_precision','val_recall','val_f1',
                  'val_loss','epochs_completed','trainable_params','train_time_sec',
                  'inference_time_sec']].round(4))
"""),
 M("## 4. 架構比較圖、訓練趨勢與混淆矩陣\n以下圖表只包含 Validation，不使用 Test。"),
 C("""import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,3.4))
ax.bar(selected.run_id,selected.val_f1)
ax.set(ylim=(0,1),ylabel='Validation F1',title='Advanced RNN vs single-layer baselines')
ax.grid(axis='y',alpha=.2)
plt.show()
for rid in wanted:
    print(rid)
    display(Image(filename=str(ROOT/f'results/figures/{rid}_learning_curve.png'),width=490))
    display(Image(filename=str(ROOT/f'results/figures/{rid}_val_confusion_matrix.png'),width=330))
"""),
 M("## 小結\nDL-07～DL-10 以共同輸入、相同 Epoch 預算進行架構比較；總參數量及訓練秒數各有差異。正式最終 Test 評估需等所有架構以 Validation 選定後才執行。")
]
save("04_IMDB_LSTM_Regularization_Matrix.ipynb",regular)
save("05_IMDB_Advanced_RNN_Architectures.ipynb",advanced)
