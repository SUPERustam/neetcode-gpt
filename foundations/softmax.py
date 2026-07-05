import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_z = np.max(z)
        shifted_z = z - max_z

        e_z = np.exp(shifted_z)
        sm = np.sum(e_z)
        return np.round(e_z / sm, 4)
