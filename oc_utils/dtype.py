import numpy as np
import torch

torch_dtype_dict = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "half": torch.float16,
    "float": torch.float32,
    "double": torch.float64,
}

torch_to_numpy_dtype_dict = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
}
