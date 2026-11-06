"""Phase 1 baseline architecture factory; reserved for later Phase 2 import."""
from __future__ import annotations
import tensorflow as tf

BASELINES={
    "DL-01": "SimpleRNN",
    "DL-02": "LSTM",
    "DL-03": "GRU",
}

def make_single_layer_rnn(run_id, *, vocab_size=12000, input_length=160,
                          embedding_dim=100, hidden_units=64, learning_rate=0.001,
                          mask_zero=True) -> tf.keras.Model:
    if run_id not in BASELINES:
        raise ValueError(f"Unknown baseline run ID: {run_id}")
    recurrent_name=BASELINES[run_id]
    cls=getattr(tf.keras.layers,recurrent_name)
    token_ids=tf.keras.Input(shape=(input_length,),dtype="int32",name="token_ids")
    embedding=tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,
                              mask_zero=mask_zero,name="embedding")(token_ids)
    encoded=cls(hidden_units,name=recurrent_name.lower())(embedding)
    result=tf.keras.layers.Dense(1,activation="sigmoid",name="positive_probability")(encoded)
    model=tf.keras.Model(token_ids,result,name=f"IMDB_{recurrent_name}_baseline")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])
    return model
