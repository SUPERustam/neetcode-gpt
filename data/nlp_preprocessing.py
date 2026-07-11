import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        words = set()
        for s in positive + negative:
            words.update(s.split())

        vocab = {word: i for i, word in enumerate(sorted(words), 1)}
        
        tensor_sequences = []
        for sen in positive + negative:
            row = [vocab[word] for word in sen.split()]
            tensor_sequences.append(torch.tensor(row))
        return nn.utils.rnn.pad_sequence(tensor_sequences, batch_first=True, padding_value=0)