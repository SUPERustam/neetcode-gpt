import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        out = np.zeros((seq_len, d_model))
        # out[::2] = np.sin((out[::2] // seq_len ) / (10000 ** (2 * (out[::2] % d_model) / d_model)))
        # out[1::2] = np.cos((out[1::2] // seq_len ) / (10000 ** (2 * (out[1::2] % d_model) / d_model)))

        pos = np.arange(seq_len)[:, None]        
        i = np.arange(0, d_model, 2)
        
        denom = 10000 ** (i / d_model)
        
        out = np.zeros((seq_len, d_model))
        
        out[:, 0::2] = np.sin(pos / denom)
        out[:, 1::2] = np.cos(pos / denom)
        
        return np.round(out, 5)
