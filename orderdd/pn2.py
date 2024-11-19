import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

data = [
    ("(x1&x2)|(x1&x2)", "x1,x2,x3"),
    ("x1&~x1", "x1,x2,x3"),
    ("(x1&(x2|x3))|(~x1&(x2|x3))", "x1,x2,x3"),
    ("(x1&x2)|(x3&x4)", "x4,x3,x2,x1"),
]

expressions, var_orders = zip(*data)

# Tokenize expressions
expr_tokenizer = Tokenizer(char_level=True, filters='', lower=False)
expr_tokenizer.fit_on_texts(expressions)
expr_sequences = expr_tokenizer.texts_to_sequences(expressions)

# Tokenize variable orders
var_tokenizer = Tokenizer(char_level=True, filters='', lower=False, split=',')
var_tokenizer.fit_on_texts(var_orders)
var_sequences = var_tokenizer.texts_to_sequences(var_orders)

# Pad sequences
expr_sequences = pad_sequences(expr_sequences, padding='post')
var_sequences = pad_sequences(var_sequences, padding='post')

# Convert to numpy arrays
X = np.array(expr_sequences)
Y = np.array(var_sequences)

def build_pointer_network(input_dim, seq_len, lstm_units):
    input_layer = Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(input_dim, lstm_units, input_length=seq_len)(input_layer)
    lstm_out = LSTM(lstm_units, return_sequences=True)(x)

    # Attention and pointer mechanism
    attention = Dense(seq_len, activation='softmax', name='attention')(lstm_out)
    pointer = tf.keras.layers.Dot(axes=[2, 1])([attention, lstm_out])

    model = Model(inputs=input_layer, outputs=pointer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Model parameters
input_dim = len(expr_tokenizer.word_index) + 1
seq_len = X.shape[1]
lstm_units = 128

model = build_pointer_network(input_dim, seq_len, lstm_units)

model.fit(X, Y, epochs=50, batch_size=16)