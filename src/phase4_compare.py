"""Validation-only Phase 4 comparison of 14 stored IMDB runs. Does not load Test."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from src.common import ROOT,file_hash
from src.evaluation import phase1_settings

ALIASES={
"ML-01-s42":"BoW + KNN","ML-02-s42":"BoW + SVC","ML-03-s42":"TF-IDF + KNN","ML-04-s42":"TF-IDF + SVC",
"DL-01-s42":"SimpleRNN","DL-02-s42":"LSTM","DL-03-s42":"GRU","DL-04-s42":"LSTM + Dropout",
"DL-05-s42":"LSTM + EarlyStopping","DL-06-s42":"LSTM + Dropout + ES",
"DL-07-s42":"Stacked LSTM","DL-08-s42":"LSTM + GRU","DL-09-s42":"Bidirectional LSTM","DL-10-s42":"Bidirectional GRU"}
P=ROOT/"config/phase4_selection_policy.yaml"
R=ROOT/"results"
F=R/"figures"

def load_and_check():
    policy=yaml.safe_load(P.read_text(encoding="utf8"))
    _,_,manifest=phase1_settings()
    assert policy["stage"]=="pre_test_selection"
    assert policy["version"]=="phase4-pretest-v1"
    assert policy["dataset_variant"]==manifest["version"]=="dedup_v1.1"
    assert policy["phase2_config_sha256"]==file_hash(ROOT/"config/phase2_conditions.yaml")
    assert policy["split_assignments_sha256"]==manifest["split_assignments_sha256"]
    assert policy["final_test_protocol"]["evaluate_now"] is False
    assert policy["selection"]["primary_score"]=="val_f1"
    assert policy["selection"]["descending_tie_break"]=="val_accuracy"
    assert policy["selection"]["final_tie_break"]=="lexicographic_run_id"
    gate=json.loads((R/"gate_c_acceptance.json").read_text())
    assert gate["status"]=="PASS" and gate["total_conditions"]==14
    data=pd.read_csv(R/"runs.csv")
    assert data.run_id.is_unique and set(data.run_id)==set(policy["eligible_runs"])==set(ALIASES)
    assert (data.status=="validation_complete").all()
    assert (data.dataset_version=="dedup_v1.1").all()
    assert (data.split_hash==manifest["split_assignments_sha256"]).all()
    assert data[["test_accuracy","test_precision","test_recall","test_f1","test_loss"]].isna().all().all()
    val=pd.read_parquet(ROOT/"data/processed/validation.parquet")
    assert len(val)==4956
    for _,row in data.iterrows():
        path=ROOT/row.predictions_path
        assert path.is_file() and (ROOT/row.model_path).is_file()
        pred=pd.read_csv(path)
        assert pred.example_id.tolist()==val.example_id.tolist()
        assert pred.label.tolist()==val.label.tolist()
        assert set(pred.prediction.unique()).issubset({0,1})
        y=pred.label.to_numpy(); yhat=pred.prediction.to_numpy()
        scores={"val_accuracy":accuracy_score(y,yhat),
                "val_precision":precision_score(y,yhat,zero_division=0),
                "val_recall":recall_score(y,yhat,zero_division=0),
                "val_f1":f1_score(y,yhat,zero_division=0)}
        for name,v in scores.items():assert abs(row[name]-v)<1e-12,(row.run_id,name)
        assert json.loads(row.confusion_matrix)==confusion_matrix(y,yhat,labels=[0,1]).tolist()
    data["family"]=np.where(data.run_id.str.startswith("ML"),"Traditional ML","Deep Learning")
    data["display_name"]=data.run_id.map(ALIASES)
    data=data.sort_values(["val_f1","val_accuracy","run_id"],
                          ascending=[False,False,True]).reset_index(drop=True)
    data.insert(0,"overall_rank",range(1,len(data)+1))
    return policy,manifest,data

def research_questions(d):
    s=d.set_index("run_id")
    f=lambda r:float(s.loc[r,"val_f1"])
    delta=lambda r,t:round(100*(f(r)-f(t)),4)
    return {
      "source":"Validation F1; signed differences in percentage points; seed=42",
      "RQ1":{"TFIDF_vs_BoW_given_KNN":delta("ML-03-s42","ML-01-s42"),
             "TFIDF_vs_BoW_given_SVC":delta("ML-04-s42","ML-02-s42")},
      "RQ2":{"SVC_vs_KNN_given_BoW":delta("ML-02-s42","ML-01-s42"),
             "SVC_vs_KNN_given_TFIDF":delta("ML-04-s42","ML-03-s42")},
      "RQ3":{"LSTM_vs_SimpleRNN":delta("DL-02-s42","DL-01-s42"),
             "GRU_vs_SimpleRNN":delta("DL-03-s42","DL-01-s42"),
             "LSTM_vs_GRU":delta("DL-02-s42","DL-03-s42")},
      "RQ4":{"Dropout_only_vs_baseline":delta("DL-04-s42","DL-02-s42"),
             "ES_only_vs_baseline":delta("DL-05-s42","DL-02-s42"),
             "Both_vs_baseline":delta("DL-06-s42","DL-02-s42"),
             "interaction_difference_in_differences":round(100*(f("DL-06-s42")-f("DL-05-s42")-f("DL-04-s42")+f("DL-02-s42")),4)},
      "RQ5":{"StackedLSTM_vs_LSTM":delta("DL-07-s42","DL-02-s42"),
             "LSTM_GRU_vs_LSTM":delta("DL-08-s42","DL-02-s42"),
             "BiLSTM_vs_LSTM":delta("DL-09-s42","DL-02-s42"),
             "BiGRU_vs_GRU":delta("DL-10-s42","DL-03-s42")},
      "caution":"One seed and one Validation split; numerical differences are not statistically significant by themselves.",
    }

def draw(d):
    F.mkdir(parents=True,exist_ok=True)
    paths=[]
    rev=d.iloc[::-1]
    fig,ax=plt.subplots(figsize=(9.3,7.1),dpi=160)
    ax.barh(rev.display_name,rev.val_f1)
    ax.set(xlim=(0,1),xlabel="Validation F1",title="IMDB — 14 conditions ranked by Validation F1")
    ax.grid(axis="x",alpha=.2);fig.tight_layout()
    p=F/"07_all_14_validation_f1.png";fig.savefig(p);plt.close(fig);paths.append(p)
    m=d.set_index("run_id")
    xs=np.arange(2); b=[m.loc[r,"val_f1"] for r in ("ML-01-s42","ML-02-s42")]
    t=[m.loc[r,"val_f1"] for r in ("ML-03-s42","ML-04-s42")]
    fig,ax=plt.subplots(figsize=(6.5,4),dpi=160)
    ax.bar(xs-.17,b,width=.33,label="BoW");ax.bar(xs+.17,t,width=.33,label="TF-IDF")
    ax.set_xticks(xs,["KNN","SVC"]);ax.set(ylim=(0,1),ylabel="Validation F1",title="RQ1 / RQ2 — 2x2 comparison")
    ax.legend();ax.grid(axis="y",alpha=.2);fig.tight_layout()
    p=F/"07_traditional_2x2_f1.png";fig.savefig(p);plt.close(fig);paths.append(p)
    rid=["DL-02-s42","DL-04-s42","DL-05-s42","DL-06-s42"]
    labels=["No Dropout\nNo ES","Dropout\nNo ES","No Dropout\nES","Dropout\nES"]
    fig,ax=plt.subplots(figsize=(7.4,4),dpi=160)
    ax.bar(labels,[m.loc[r,"val_f1"] for r in rid])
    ax.set(ylim=(0,1),ylabel="Validation F1",title="RQ4 — LSTM regularization 2x2")
    ax.grid(axis="y",alpha=.2);fig.tight_layout()
    p=F/"07_regularization_2x2_f1.png";fig.savefig(p);plt.close(fig);paths.append(p)
    aid=["DL-02-s42","DL-03-s42","DL-07-s42","DL-08-s42","DL-09-s42","DL-10-s42"]
    fig,ax=plt.subplots(figsize=(10.5,4.3),dpi=160)
    ax.bar([ALIASES[r] for r in aid],[m.loc[r,"val_f1"] for r in aid])
    ax.set(ylim=(0,1),ylabel="Validation F1",title="RQ3 / RQ5 — Recurrent models")
    ax.tick_params(axis="x",rotation=17);ax.grid(axis="y",alpha=.2);fig.tight_layout()
    p=F/"07_advanced_rnn_f1.png";fig.savefig(p);plt.close(fig);paths.append(p)
    dl=d[d.family=="Deep Learning"]
    fig,ax=plt.subplots(figsize=(8,4.5),dpi=160)
    ax.scatter(dl.train_time_sec,dl.val_f1,s=50)
    for _,r in dl.iterrows():
        ax.annotate(r.run_id[:5],(r.train_time_sec,r.val_f1),
                    xytext=(4,4),textcoords="offset points",fontsize=8)
    ax.set(xlabel="DL fit seconds (includes per-epoch Validation)",
           ylabel="Validation F1",title="Deep learning: observed time versus F1")
    ax.grid(alpha=.2);fig.tight_layout()
    p=F/"07_deep_learning_cost_f1.png";fig.savefig(p);plt.close(fig);paths.append(p)
    return paths

def generate():
    policy,manifest,d=load_and_check()
    rq=research_questions(d)
    rank=R/"phase4_ranked_validation.csv"
    fields=["overall_rank","run_id","family","display_name","val_accuracy","val_precision",
            "val_recall","val_f1","val_loss","trainable_params",
            "train_time_sec","inference_time_sec","epochs_completed","model_path","predictions_path"]
    d[fields].to_csv(rank,index=False,float_format="%.12g",lineterminator="\n")
    (R/"phase4_research_questions.json").write_text(json.dumps(rq,ensure_ascii=False,indent=2)+"\n")
    figures=draw(d)
    primary=d.iloc[0].run_id
    ml=d[d.family=="Traditional ML"].iloc[0].run_id
    dl=d[d.family=="Deep Learning"].iloc[0].run_id
    choice={"overall":primary,"best_ml":ml,"best_dl":dl,"locked_finalists":[ml,dl]}
    manifest_out={"stage":"pre_test","test_evaluations":0,"dataset":"dedup_v1.1",
        "eligible_models":14,"policy_sha256":file_hash(P),
        "runs_sha256":file_hash(R/"runs.csv"),
        "ranking_sha256":file_hash(rank),
        "rq_sha256":file_hash(R/"phase4_research_questions.json"),
        "choices":choice,
        "figures":[str(p.relative_to(ROOT)) for p in figures]}
    (R/"phase4_validation_summary.json").write_text(json.dumps(manifest_out,ensure_ascii=False,indent=2)+"\n")
    lines=["# IMDB NLP 矩陣實驗 — Phase 4 / Test 前模型比較",
      "","**已完成 14 組 Validation 統整與模型選擇候選；此處不是 Gate E，未使用官方 Test。**",
      "","## 14 組 Validation 排名（F1 由高至低，平手以 Accuracy / Run ID）",
      "| 排名 | Run | 模型 | Accuracy | Precision | Recall | F1 | Params | Fit 秒 | 推論秒 |",
      "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _,r in d.iterrows():
        pa="—" if pd.isna(r.trainable_params) else f"{int(r.trainable_params):,}"
        lines.append(f"| {int(r.overall_rank)} | {r.run_id} | {r.display_name} | {r.val_accuracy:.2%} | {r.val_precision:.2%} | {r.val_recall:.2%} | {r.val_f1:.2%} | {pa} | {r.train_time_sec:.2f} | {r.inference_time_sec:.2f} |")
    lines+=["","## Test 前候選（不根據 Test 調整）",
       f"- 全組最高：**{primary}（{ALIASES[primary]}）**。",
       f"- 傳統機器學習代表：**{ml}（{ALIASES[ml]}）**。",
       f"- 深度學習代表：**{dl}（{ALIASES[dl]}）**。",
       "- 進入下一階段前，將以 model_selection_lock.json 鎖定兩個類別代表；建議官方 Test 只評估這兩組各一次。",
       "","## 研究問題結論（Validation F1 百分點差；單次 seed=42）"]
    for rqid,values in rq.items():
        if not rqid.startswith("RQ"):continue
        lines+=["",f"### {rqid}"]
        lines+=[f"- {k}: {v:+.2f} 百分點" for k,v in values.items()]
    lines+=["","## 圖表",
        *[f"- {p.relative_to(ROOT)}" for p in figures],
        "- 既有混淆矩陣：每模型 1 張；既有 Loss curves：每個 DL 模型 1 張。Notebook 07 直接引用，不改舊結果。",
        "","## 評估限制",
        "- 全部都是 Validation；不能當作獨立 Test 泛化成績。",
        "- 單一 seed 42，差值未進行顯著性檢定。早停組實際 Epoch 不同。",
        "- 模型參數量並非匹配；GPU/CPU 訓練時間計算範圍也不相同。",
        "- KNN fit 時間不包含昂貴的距離查詢；SVC fit 秒數含兩個候選 C 的訓練；DL 含每個 Epoch 的 Validation。",
        "- 160-token 輸入截斷超過半數評論；官方 Test 內部重複保留；未稽核語意近似重複。",
        "- 選做 GloVe（TL-01／TL-02）未訓練，不列入 14 個必做條件。",
        "","## Gate E 尚待完成",
        "先凍結 selection lock，再進行事先決定範圍的單次官方 Test 評估；不能看 Test 再改模型、選參數或改變正式候選。"]
    report=R/"PHASE4_PRETEST_REPORT.md"
    report.write_text("\n".join(lines)+"\n",encoding="utf8")
    print("PHASE4_VALIDATION_COMPARISON_OK",len(d),"Test_used=0")
    print("MODEL_CHOICES",choice)
    print("RQ",json.dumps(rq,ensure_ascii=False))
    print("OUTPUTS",rank,report,len(figures),"figures")

if __name__=="__main__":
    generate()
