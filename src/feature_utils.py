"""Train-only BoW / TF-IDF / Keras TextVectorization artifacts for dedup_v1.1."""
from __future__ import annotations
import json

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import tensorflow as tf
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.common import ROOT, config, file_hash, load_splits


def prepare_features() -> dict:
    cfg,c_hash=config()
    manifest=json.loads((ROOT/"data/processed/split_manifest.json").read_text(encoding="utf-8"))
    assert cfg["version"]==manifest["version"]=="dedup_v1.1"
    assert c_hash==manifest["config_sha256"]
    frames=load_splits()
    for n,df in frames.items():
        assert file_hash(ROOT/f"data/processed/{n}.parquet")==manifest["counts"][n]["sha256"]
        assert df.example_id.is_unique and df.cleaned_text.str.len().gt(0).all()
    assert not set(frames["train"].cleaned_hash)&set(frames["validation"].cleaned_hash)
    assert not set(frames["train"].cleaned_hash)&set(frames["test"].cleaned_hash)
    assert not set(frames["validation"].cleaned_hash)&set(frames["test"].cleaned_hash)
    art=ROOT/"artifacts/vectorizers"
    art.mkdir(parents=True,exist_ok=True)
    ml=cfg["traditional"]
    opts={
        "max_features":int(ml["max_features"]),
        "ngram_range":tuple(ml["ngram_range"]),
        "min_df":int(ml["min_df"]),
        "lowercase":False,
        "token_pattern":ml["token_pattern"],
        "stop_words":None,
    }
    bow=CountVectorizer(**opts,dtype=np.int32)
    tfidf=TfidfVectorizer(**opts,dtype=np.float64,
                          norm=ml["tfidf_norm"],use_idf=bool(ml["use_idf"]))
    train_text=frames["train"].cleaned_text.tolist()
    bow_train=bow.fit_transform(train_text)
    tfidf_train=tfidf.fit_transform(train_text)
    assert bow_train.shape[0]==tfidf_train.shape[0]==cfg["expected_v1_1"]["train"]
    assert len(bow.get_feature_names_out())==len(tfidf.get_feature_names_out())==12000
    paths={}
    for name,v in (("bow_count_vectorizer.joblib",bow),("tfidf_vectorizer.joblib",tfidf)):
        dest=art/name
        joblib.dump(v,dest)
        paths[name]=file_hash(dest)
    matrix_meta={}
    for name,vec,train_matrix in (("bow",bow,bow_train),("tfidf",tfidf,tfidf_train)):
        for split,frame in frames.items():
            matrix=(train_matrix if split=="train" else vec.transform(frame.cleaned_text.tolist()))
            dest=art/f"{name}_{split}.npz"
            save_npz(dest,matrix,compressed=True)
            matrix_meta[f"{name}_{split}"]={
                "rows":int(matrix.shape[0]),"features":int(matrix.shape[1]),
                "nnz":int(matrix.nnz),"sha256":file_hash(dest),
            }

    t=cfg["text_vectorization"]
    layer=tf.keras.layers.TextVectorization(
         max_tokens=int(t["max_tokens"]),standardize=None,
         split=t["split"],output_mode=t["output_mode"],
         output_sequence_length=int(t["output_sequence_length"]),
    )
    ds=tf.data.Dataset.from_tensor_slices(train_text).batch(int(t["adapt_batch_size"]))
    layer.adapt(ds)  # train ONLY
    vocab=layer.get_vocabulary()
    assert len(vocab)==12000 and vocab[:2]==["","[UNK]"]
    vocab_path=art/"keras_vocabulary_12000.txt"
    vocab_path.write_text("\n".join(vocab)+"\n",encoding="utf-8")
    assert len(set(vocab))==len(vocab)
    encoded={}
    lenstats={}
    vocab_set=set(vocab)
    for name,frame in frames.items():
        reviews=frame.cleaned_text.tolist()
        batches=[
            layer(tf.constant(reviews[i:i+512])).numpy().astype(np.int32)
            for i in range(0,len(reviews),512)
        ]
        encoded[name]=np.concatenate(batches,axis=0)
        assert encoded[name].shape==(len(frame),160)
        lengths=np.fromiter((len(x.split()) for x in reviews),dtype=np.int32,count=len(reviews))
        oov=sum(sum(token not in vocab_set for token in review.split()) for review in reviews)
        lenstats[name]={
            "rows":len(frame),
            "mean_token_count":round(float(lengths.mean()),3),
            "truncated_records":int(np.count_nonzero(lengths>160)),
            "truncated_fraction":round(float(np.mean(lengths>160)),6),
            "oov_token_fraction_before_truncation":round(oov/max(int(lengths.sum()),1),6),
        }
    arrays={}
    for name,frame in frames.items():
        arrays[f"X_{name}"]=encoded[name]
        arrays[f"y_{name}"]=frame.label.to_numpy(dtype=np.int8)
        arrays[f"ids_{name}"]=frame.example_id.to_numpy(dtype="U18")
    array_path=art/"keras_token_ids_12000_160.npz"
    np.savez_compressed(array_path,**arrays)
    meta={
        "version":"dedup_v1.1",
        "config_sha256":c_hash,
        "manifest_split_sha256":manifest["split_assignments_sha256"],
        "preprocessing_version":cfg["preprocessing"]["version"],
        "fitted_on":"dedup_v1.1 train only",
        "fit_policy":{"bow":"fit_transform(train)", "tfidf":"fit_transform(train)", "keras":"adapt(train)"},
        "validation_test_policy":{"bow":"transform only","tfidf":"transform only","keras":"inference only"},
        "features":{"bow":len(bow.vocabulary_), "tfidf":len(tfidf.vocabulary_), "keras":len(vocab)},
        "vectorizers_sha256":paths,
        "sparse_matrices":matrix_meta,
        "keras_vocabulary_sha256":file_hash(vocab_path),
        "keras_token_ids_sha256":file_hash(array_path),
        "sequence_length":160,
        "lengths":lenstats,
        "versions":{"tensorflow":tf.__version__,"scikit_learn":sklearn.__version__,
                    "scipy":scipy.__version__,"numpy":np.__version__,"pandas":pd.__version__},
    }
    (ROOT/"results/feature_metadata.json").write_text(
        json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return meta
