"""GloVe 6B 100d from a verified Zenodo word2vec-binary archive.

Word2Vec binary is a serialization of pretrained GloVe vectors, not a new
embedding training algorithm. Align exact tokens to the frozen Train-only
Keras TextVectorization vocabulary; do not use Validation or Test statistics.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import yaml

from src.common import ROOT, file_hash

OPT=ROOT/"optional_glove"
SRC=OPT/"source"
ART=OPT/"artifacts"
RESULTS=OPT/"results"
CONFIG=OPT/"experiment.yaml"

def digest(path,algorithm):
    h=hashlib.new(algorithm)
    with path.open("rb") as f:
        for blob in iter(lambda:f.read(8*1024*1024),b""):
            h.update(blob)
    return h.hexdigest()

def freeze_official():
    target=RESULTS/"FROZEN_OFFICIAL_GATE_E_BEFORE_OPTIONAL.json"
    if target.is_file():
        data=json.loads(target.read_text(encoding="utf8"))
        assert all(file_hash(ROOT/p)==sha for p,sha in data["source_artifacts_sha256"].items())
        return data
    lock=json.loads((ROOT/"results/model_selection_lock.json").read_text())
    gate=json.loads((ROOT/"results/gate_e_acceptance.json").read_text())
    assert lock["status"]=="LOCKED_BEFORE_FINAL_TEST" and gate["status"]=="PASS"
    files=[
       "results/runs.csv","results/model_selection_lock.json",
       "results/gate_c_acceptance.json","results/gate_e_acceptance.json",
       "results/GATE_E_REPORT.md","notebooks/07_IMDB_Results_Comparison.ipynb",
       "data/processed/split_manifest.json",
       "artifacts/vectorizers/keras_vocabulary_12000.txt",
       "artifacts/vectorizers/keras_token_ids_12000_160.npz",
       "results/final_test/locked_final_test_metrics.json",
       "results/final_test/ML-04-s42_test_predictions.csv",
       "results/final_test/DL-05-s42_test_predictions.csv",
    ]
    snapshot={name:file_hash(ROOT/name) for name in files}
    for rid,item in lock["source_artifacts"].items():
        assert file_hash(ROOT/item["model_path"])==item["model_sha256"]
        snapshot[item["model_path"]]=item["model_sha256"]
    value={"status":"FROZEN_BEFORE_GLOVE","original_gate_e":"PASS",
           "official_14_runs":14,"comparison":"Validation-only post Gate E",
           "source_artifacts_sha256":snapshot}
    with target.open("x",encoding="utf8") as f:
        f.write(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    print("FROZEN_GATE_E",len(snapshot),"files",flush=True)
    return value

def parse_binary(binary,targets,matrix,matched,dim):
    with binary.open("rb",buffering=4*1024*1024) as f:
        header=f.readline().strip().split()
        assert len(header)==2 and header==[b"400000",b"100"],header
        total=int(header[0])
        for i in range(total):
            word=bytearray()
            while True:
                ch=f.read(1)
                if ch==b" ":break
                if ch in (b"",b"\n"):
                    raise ValueError(f"Corrupted binary word at {i}")
                word+=ch
                assert len(word)<1500
            raw=f.read(dim*4)
            assert len(raw)==dim*4,f"Truncated vector {i}"
            idx=targets.get(word.decode("utf8",errors="replace"))
            if idx is not None:
                vec=np.frombuffer(raw,dtype="<f4")
                assert np.isfinite(vec).all()
                matrix[idx,:]=vec
                matched[idx]=True
            suffix=f.read(1)
            # The Zenodo word2vec-binary file concatenates some entries without
            # a newline separator (e.g., the next token is a comma).  Rewind
            # the first byte of the next word instead of dropping it.
            if suffix not in (b"\n",b""):
                f.seek(-1,1)
        assert f.read(1)==b"","Unexpected extra binary data"
    return total

def main():
    c=yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    assert c["version"]=="optional-glove-v1.0"
    assert c["comparison"]["official_Test_evaluation"]=="prohibited"
    assert c["baseline_run"]=="DL-03-s42"
    src=c["source"]
    assert src["format"]=="word2vec_binary_float32"
    assert src["expected_binary_bytes"]==163356485
    assert c["common_training"]["embedding_dim"]==100
    for folder in (SRC,ART,RESULTS):folder.mkdir(parents=True,exist_ok=True)
    freeze_official()
    archive=SRC/src["archive_name"]
    assert archive.is_file()
    assert digest(archive,"md5")==src["expected_archive_md5"],"Zenodo GloVe ZIP MD5 mismatch"
    binary=SRC/src["binary_name"]
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert names==[src["binary_name"]],names
        info=z.getinfo(src["binary_name"])
        assert info.file_size==src["expected_binary_bytes"]
        if not binary.exists():
            part=binary.with_name(binary.name+".part")
            assert not part.exists(),"Partial extraction found: inspect manually"
            with z.open(info) as fi,part.open("xb") as fo:
                shutil.copyfileobj(fi,fo,length=2*1024*1024)
            part.rename(binary)
    assert binary.stat().st_size==163356485
    vpath=ROOT/c["embedding_preparation"]["vocab_source"]
    vocab=vpath.read_text(encoding="utf8").splitlines()
    assert len(vocab)==12000 and vocab[:2]==["","[UNK]"]
    assert len(set(vocab))==12000
    dim=100
    rng=np.random.default_rng(42)
    matrix=rng.normal(0,0.05,size=(12000,dim)).astype(np.float32)
    matrix[0,:]=0
    matched=np.zeros(12000,dtype=bool)
    needed={v:i for i,v in enumerate(vocab) if i>=2}
    count=parse_binary(binary,needed,matrix,matched,dim)
    assert count==400000
    assert matched.sum()>5000
    assert matched[0]==False and matched[1]==False
    assert np.all(matrix[0]==0) and np.isfinite(matrix).all()
    with np.load(ROOT/c["embedding_preparation"]["token_id_source"],allow_pickle=False) as z:
        X=z["X_train"]
    assert X.shape==(19823,160)
    freqs=np.bincount(X.ravel(),minlength=12000)
    known=int(freqs[2:].sum())
    hits=int(freqs[matched].sum())
    nonpad=int(freqs[1:].sum())
    coverage={
       "source":"GloVe 6B 100d Zenodo word2vec binary converted from pretrained GloVe",
       "source_rows":count,"vector_dim":dim,
       "train_fitted_vocab_size":len(vocab),
       "eligible_vocab_excluding_PAD_UNK":len(vocab)-2,
       "exact_matching_vocab":int(matched.sum()),
       "unmatched_vocab":int(11998-matched.sum()),
       "vocab_exact_match_fraction":float(matched.sum()/11998),
       "train_nonpad_known_token_occurrences":known,
       "train_nonpad_unk_occurrences":int(freqs[1]),
       "train_glove_matched_token_occurrences":hits,
       "train_coverage_excluding_UNK":float(hits/known),
       "train_coverage_including_UNK":float(hits/nonpad),
       "pad_index":0,"unk_index":1,"pad_all_zero":True,
       "unk_init":"Gaussian random 0.05, seed42",
       "no_glove_match_init":"Gaussian random 0.05, seed42",
       "fit_from_Train_only":True,"validation_or_test_statistics_used":False,
       "notes":"Exact-token alignment only; punctuation-attached vocabulary tokens can miss GloVe.",
    }
    matfile=ART/"glove_6B_100d_aligned_matrix.npy"
    maskfile=ART/"glove_6B_100d_matched_mask.npy"
    assert not matfile.exists() and not maskfile.exists(),"Refuse overwrite GloVe initialization"
    np.save(matfile,matrix,allow_pickle=False)
    np.save(maskfile,matched,allow_pickle=False)
    evidence={
      "canonical_glove_project":src["original_project"],
      "download_mirror":src["download_mirror"],
      "source_format":"word2vec_binary_float32",
      "source_word_count":count,
      "license":src["license"],
      "zip_path":str(archive.relative_to(ROOT)),
      "zip_size_bytes":archive.stat().st_size,
      "zip_md5":digest(archive,"md5"),
      "zip_sha256":file_hash(archive),
      "glove_binary_path":str(binary.relative_to(ROOT)),
      "glove_binary_size_bytes":binary.stat().st_size,
      "glove_binary_sha256":file_hash(binary),
      "aligned_matrix_path":str(matfile.relative_to(ROOT)),
      "aligned_matrix_sha256":file_hash(matfile),
      "aligned_mask_path":str(maskfile.relative_to(ROOT)),
      "aligned_mask_sha256":file_hash(maskfile),
      "experiment_config_sha256":file_hash(CONFIG),
      "vocabulary_path":str(vpath.relative_to(ROOT)),
      "vocabulary_sha256":file_hash(vpath),
      "train_token_id_source_sha256":file_hash(ROOT/c["embedding_preparation"]["token_id_source"]),
    }
    (RESULTS/"glove_coverage.json").write_text(json.dumps(coverage,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    (RESULTS/"glove_source_manifest.json").write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("GLOVE_ALIGNED_PASS","matched_vocab",coverage["exact_matching_vocab"],
          "of",11998,"vocab_percent",round(coverage["vocab_exact_match_fraction"]*100,2),
          "train_token_percent",round(coverage["train_coverage_excluding_UNK"]*100,2),flush=True)
    print("PROVENANCE_SHA",evidence["glove_binary_sha256"],"MATRIX_SHA",evidence["aligned_matrix_sha256"],flush=True)

if __name__=="__main__":
    main()
