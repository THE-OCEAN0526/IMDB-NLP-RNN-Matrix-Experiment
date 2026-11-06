"""Approved v1.1: preserve official Test; hash full cleaned reviews; deduplicate Train."""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src.common import ROOT, clean_review, config, file_hash, ids_hash, review_hash, load_splits

SPLITS = ("train", "validation", "test")

def source_frame(source: pd.DataFrame, name: str) -> pd.DataFrame:
    assert list(source.columns) == ["text", "label"]
    assert source[["text", "label"]].notna().all().all()
    assert set(source.label.unique()) == {0,1}
    idx = np.arange(len(source), dtype=np.int32)
    raw = source["text"].astype(str).tolist()
    cleaned = [clean_review(x) for x in raw]
    assert all(cleaned)
    return pd.DataFrame({
        "example_id": [f"imdb-{name}-{int(x):05d}" for x in idx],
        "source_split": name,
        "source_index": idx,
        "text": raw,
        "cleaned_text": cleaned,
        "cleaned_hash": [review_hash(x) for x in cleaned],
        "label": source["label"].to_numpy(dtype=np.int8),
    })

def exclusion_ledger(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    out = df[["example_id","source_split","source_index","label","cleaned_hash"]].copy()
    out["reason"] = reason
    return out

def overlaps(datasets: dict[str,pd.DataFrame]) -> dict[str,int]:
    hashes = {k: set(v.cleaned_hash) for k,v in datasets.items()}
    return {
       "train_validation":len(hashes["train"]&hashes["validation"]),
       "train_test":len(hashes["train"]&hashes["test"]),
       "validation_test":len(hashes["validation"]&hashes["test"]),
    }

def prepare() -> dict:
    cfg, cfg_hash = config()
    assert cfg["version"] == "dedup_v1.1"
    assert cfg["preprocessing"]["version"] == "phase0-text-v1"
    assert cfg["dedup"]["text_type"] == "full_cleaned_text"
    assert cfg["dedup"]["abort_on_label_conflict"] is True
    assert cfg["dataset"]["preserve_official_test"] is True
    r = ROOT/"data/raw"
    p = ROOT/"data/processed"
    outdir = ROOT/"results"
    for directory in (r,p,outdir):
        directory.mkdir(parents=True,exist_ok=True)
    originals = {}
    for name in ("train","test"):
        source = Path(cfg["dataset"][f"source_{name}"])
        expected_hash=cfg["dataset"][f"sha256_{name}"]
        assert source.is_file() and file_hash(source)==expected_hash, f"Source SHA mismatch: {name}"
        dest=r/(name+".parquet")
        if not dest.exists():
            shutil.copyfile(source,dest)  # keep original file untouched
        assert file_hash(dest)==expected_hash, f"Local raw SHA mismatch: {name}"
        originals[name]=pd.read_parquet(dest)
        assert len(originals[name])==cfg["dataset"][f"{name}_source_rows"]
        assert originals[name].label.value_counts().to_dict()=={0:12500,1:12500}
    pool=source_frame(originals["train"], "train")
    official_test=source_frame(originals["test"], "test")
    # A label contradiction across identical cleaned text is a blocking data error.
    all_hash_labels=pd.concat([pool[["cleaned_hash","label"]],official_test[["cleaned_hash","label"]]],
                              ignore_index=True).groupby("cleaned_hash").label.nunique()
    contradictions=all_hash_labels[all_hash_labels>1]
    assert len(contradictions)==0, f"Conflicting labels for {len(contradictions)} identical cleaned reviews"
    test_hashes=set(official_test.cleaned_hash)
    exclude_test=pool.loc[pool.cleaned_hash.isin(test_hashes)].copy()
    eligible=pool.loc[~pool.cleaned_hash.isin(test_hashes)].copy()
    assert eligible.source_index.is_monotonic_increasing
    duplicated=eligible.cleaned_hash.duplicated(keep="first")
    exclude_within=eligible.loc[duplicated].copy()
    unique=eligible.loc[~duplicated].copy().reset_index(drop=True)
    assert unique.cleaned_hash.is_unique
    ids=np.arange(len(unique))
    tr_ix,val_ix=train_test_split(ids,
          test_size=cfg["dataset"]["val_fraction"],
          random_state=cfg["dataset"]["seed"],
          stratify=unique.label.to_numpy())
    datasets={
        "train":unique.iloc[np.sort(tr_ix)].reset_index(drop=True),
        "validation":unique.iloc[np.sort(val_ix)].reset_index(drop=True),
        "test":official_test.reset_index(drop=True),
    }
    expected=cfg["expected_v1_1"]
    assert len(exclude_test)==expected["exclude_test_overlap"]==123
    assert len(exclude_within)==expected["exclude_internal_duplicates"]==98
    assert all(len(datasets[n])==expected[n] for n in SPLITS)
    assert len(datasets["train"])+len(datasets["validation"])+len(exclude_test)+len(exclude_within)==25000
    assert all(value==0 for value in overlaps(datasets).values())
    assert len(set(datasets["train"].example_id)&set(datasets["validation"].example_id))==0
    assert ids_hash(datasets["train"].example_id.tolist())==expected["train_ids_sha256"]
    assert ids_hash(datasets["validation"].example_id.tolist())==expected["validation_ids_sha256"]
    assert ids_hash(exclude_test.example_id.tolist())==expected["excluded_test_overlap_ids_sha256"]
    assert ids_hash(exclude_within.example_id.tolist())==expected["excluded_internal_ids_sha256"]

    for n,frame in datasets.items():
        frame.to_parquet(p/f"{n}.parquet",index=False)
    logs={
       "excluded_test_overlap.csv": exclusion_ledger(exclude_test,"same_cleaned_text_as_official_test"),
       "excluded_intra_train_duplicates.csv": exclusion_ledger(exclude_within,"duplicate_of_earlier_training_source_index"),
    }
    for name,df in logs.items():
        df.to_csv(p/name,index=False,lineterminator="\n")
    selections=pd.concat(
        [datasets[n][["example_id","source_split","source_index","label","cleaned_hash"]].assign(split=n)
         for n in SPLITS],ignore_index=True)
    selections.to_csv(p/"split_assignments.csv",index=False,lineterminator="\n")
    assert selections.example_id.is_unique
    assert len(selections)+len(exclude_test)+len(exclude_within)==50000
    # Strict source Test equivalence despite adding audit columns to the derived copy.
    assert datasets["test"].text.tolist()==originals["test"].text.tolist()
    assert datasets["test"].label.tolist()==originals["test"].label.tolist()
    counts={n:{
       "rows":len(datasets[n]),
       "labels":{str(k):int(v) for k,v in datasets[n].label.value_counts().sort_index().items()},
       "sha256":file_hash(p/f"{n}.parquet"),
       "source_ids_sha256":ids_hash(datasets[n].example_id.tolist()),
       "unique_cleaned_text":int(datasets[n].cleaned_hash.nunique()),
    } for n in SPLITS}
    audit={
       "version":"dedup_v1.1",
       "source_train_rows":25000,
       "official_test_rows":25000,
       "excluded_test_overlap":{"rows":len(exclude_test),"labels":{str(k):int(v) for k,v in exclude_test.label.value_counts().sort_index().items()}},
       "excluded_intra_train_duplicates":{"rows":len(exclude_within),"labels":{str(k):int(v) for k,v in exclude_within.label.value_counts().sort_index().items()}},
       "train_val_test_cross_split_cleaned_text_overlaps":overlaps(datasets),
       "test_internal_duplicate_groups":int(official_test.groupby("cleaned_hash").size().gt(1).sum()),
       "test_internal_extra_duplicate_rows":int(len(official_test)-official_test.cleaned_hash.nunique()),
       "global_conflicting_label_hash_groups":0,
       "warning":"No cross-split identical full cleaned text; semantic near-duplicates not tested and official Test internal duplicates retained.",
    }
    (outdir/"data_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    manifest={
       "dataset":"Stanford IMDB (stanfordnlp/imdb)",
       "version":"dedup_v1.1",
       "preprocessing_version":cfg["preprocessing"]["version"],
       "config_sha256":cfg_hash,
       "source_raw_sha256":{"train":cfg["dataset"]["sha256_train"],"test":cfg["dataset"]["sha256_test"]},
       "local_raw_sha256":{n:file_hash(r/(n+".parquet")) for n in ("train","test")},
       "seed":cfg["dataset"]["seed"],
       "test_size":cfg["dataset"]["val_fraction"],
       "stratify":"label",
       "dedup":cfg["dedup"],
       "counts":counts,
       "excluded_lists_sha256":{n:file_hash(p/n) for n in logs},
       "split_assignments_sha256":file_hash(p/"split_assignments.csv"),
       "audit":audit,
       "versions":{"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__},
    }
    (p/"split_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return manifest
