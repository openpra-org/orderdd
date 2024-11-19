# Required Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

if __name__ == "__main__":
    # Read the CSV data
    data = pd.read_csv('../orderdd/datasets/inputs.csv')  # 131,072

    # Data processing
    expressions = data['expression'].tolist()
    reorderings = data['reordering'].tolist()