"""Phase 2 architecture factory, preserving Phase 1 Embedding/Adam/BCE settings."""
from __future__ import annotations
import tensorflow as tf

ADVANCED={
    "DL-07":"StackedLSTM",
    "DL-08":"LSTM_GRU",
    "DL-09":"BidirectionalLSTM",
    "DL-10":"BidirectionalGRU",
}

def build_phase2(run_id,shared,dropout_fraction=0.3):
    token=tf.keras.Input(shape=(int(shared["sequence_length"]),),
                         dtype="int32",name="token_ids")
    x=tf.keras.layers.Embedding(input_dim=int(shared["vocab_size"]),
           output_dim=int(shared["embedding_dim"]),
           mask_zero=bool(shared["mask_zero"]),name="embedding")(token)
    units=int(shared["hidden_units"])
    if run_id in ("DL-04","DL-05","DL-06"):
        x=tf.keras.layers.LSTM(units,name="lstm")(x)
        if run_id in ("DL-04","DL-06"):
            x=tf.keras.layers.Dropout(dropout_fraction,name="dropout")(x)
    elif run_id=="DL-07":
        x=tf.keras.layers.LSTM(units,return_sequences=True,name="lstm_1")(x)
        x=tf.keras.layers.LSTM(units,name="lstm_2")(x)
    elif run_id=="DL-08":
        x=tf.keras.layers.LSTM(units,return_sequences=True,name="lstm_1")(x)
        x=tf.keras.layers.GRU(units,name="gru_2")(x)
    elif run_id=="DL-09":
        x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units),
                                name="bidirectional_lstm")(x)
    elif run_id=="DL-10":
        x=tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units),
                                name="bidirectional_gru")(x)
    else:
        raise ValueError("Unsupported Phase 2 Run ID: "+str(run_id))
    output=tf.keras.layers.Dense(1,activation="sigmoid",
                                name="positive_probability")(x)
    m=tf.keras.Model(inputs=token,outputs=output,name=f"IMDB_{run_id.replace('-','')}")
    m.compile(optimizer=tf.keras.optimizers.Adam(float(shared["learning_rate"])),
              loss="binary_crossentropy",
              metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])
    return m
