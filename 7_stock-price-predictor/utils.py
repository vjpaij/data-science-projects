"""
This module includes utility functions for preprocessing input sequences before feeding them into the model.

preprocess_input: Takes a raw sequence of stock prices, scales it using MinMaxScaler, and reshapes it into a tensor suitable for 
the LSTM model.
"""


import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

def preprocess_input(sequence, sequence_length=60):
    """
    Preprocesses the input sequence for the LSTM model.

    Parameters:
    - sequence (list or array): The raw input sequence of stock prices.
    - sequence_length (int): The length of the sequence to be used for prediction.

    Returns:
    - torch.Tensor: Preprocessed tensor ready for model input.
    """
    # Convert the sequence to a NumPy array and reshape
    sequence = np.array(sequence).reshape(-1, 1)

    # Scale the sequence
    scaled_sequence = scaler.fit_transform(sequence)

    # Ensure the sequence has the required length
    if len(scaled_sequence) < sequence_length:
        raise ValueError(f"Input sequence must be at least {sequence_length} in length.")

    # Extract the last 'sequence_length' elements
    input_seq = scaled_sequence[-sequence_length:]

    # Convert to tensor and reshape for model input
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
    return input_tensor
