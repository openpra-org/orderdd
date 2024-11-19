import tensorflow as tf


class PointerNetwork(tf.keras.Model):
    def __init__(self, units):
        super(PointerNetwork, self).__init__()
        self.encoder = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.pointer = tf.keras.layers.Dense(1, activation='softmax')

    def call(self, inputs, score):
        # Expand and repeat score to concatenate with inputs
        score_expanded = tf.expand_dims(tf.expand_dims(score, -1), -1)
        score_repeated = tf.repeat(score_expanded, repeats=inputs.shape[1], axis=1)

        # Concatenate score with inputs
        inputs_with_score = tf.concat([inputs, score_repeated], axis=-1)

        encoder_out, enc_hidden, enc_cell = self.encoder(inputs_with_score)
        dec_hidden, dec_cell = enc_hidden, enc_cell
        dec_input = tf.expand_dims(tf.reduce_mean(inputs_with_score, axis=1), 1)

        outputs = []
        for _ in range(inputs.shape[1]):
            dec_out, dec_hidden, dec_cell = self.decoder(dec_input, initial_state=[dec_hidden, dec_cell])
            logits = self.pointer(tf.keras.layers.dot([dec_out, encoder_out], axes=[2, 2]))
            dec_input = tf.matmul(logits, encoder_out, transpose_a=True)
            outputs.append(logits)

        return tf.concat(outputs, axis=1)


def generate_data(batch_size, sequence_len, vector_dim):
    while True:
        yield tf.random.uniform((batch_size, sequence_len, vector_dim))

dataset = tf.data.Dataset.from_generator(
    generate_data, args=[100, 5, 2], output_signature=tf.TensorSpec(shape=(100, 5, 2), dtype=tf.float32)
)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
