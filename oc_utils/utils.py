import mlflow
import dataclasses
import numpy as np
from einops import pack, unpack
from pathlib import Path
from typing import Union

import torch

from configs.config import Config

REPO_PATH = repo_path = Path(__file__).resolve().parents[1]
FloatTensor = Union[torch.FloatTensor, torch.HalfTensor]


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flattens a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): Key for the current level in recursion (default is empty).
        sep (str): Separator for joining keys (default is ".").

    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_params(config: Config):
    config_dict = dataclasses.asdict(config)
    config_dict = {k: v for k, v in config_dict.items() if not k.endswith("_all")}
    config_dict = flatten_dict(config_dict)
    mlflow.log_params(config_dict)


def convert_state_dict_dtype(state_dict, dtype):
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = value.to(dtype)
        elif isinstance(value, dict):  # if it's a nested dict
            state_dict[key] = convert_state_dict_dtype(value, dtype)
    return state_dict


def to_tensor(x, device, **kwargs):
    if x is None:
        return None
    elif isinstance(x, np.ndarray):
        return torch.tensor(x, device=device)
    elif isinstance(x, torch.Tensor):
        return x.to(device=device, **kwargs)
    elif isinstance(x, dict):
        return {k: to_tensor(v, device, **kwargs) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)([to_tensor(v, device, **kwargs) for v in x])
    elif dataclasses.is_dataclass(x):
        return dataclasses.replace(x, **{k: to_tensor(v, device, **kwargs) for k, v in dataclasses.asdict(x).items()})
    else:
        raise ValueError(f"unsupported type: {type(x)}")


def are_dicts_equal(
    dict1, dict2,
    keys=(),
    exclude_keys=(),
    only_use_common_keys=False,
):
    # Compare dict keys
    if not keys:
        if only_use_common_keys:
            keys = set(dict1.keys()) & set(dict2.keys())
        else:
            keys = set(dict1.keys()) | set(dict2.keys())
    else:
        assert not exclude_keys, "exclude_keys is not supported when keys is specified."
        assert not only_use_common_keys, "only_use_common_keys is not supported when keys is specified."

    for key in keys:
        if key in exclude_keys:
            continue

        if key not in dict1:
            print(f"Key {key} not found in dict1.")
            return False

        if key not in dict2:
            print(f"Key {key} not found in dict2.")
            return False

    # Compare dict values
    for key in keys:
        if key in exclude_keys:
            continue

        value1, value2 = dict1[key], dict2[key]

        if isinstance(value1, torch.Tensor):
            assert isinstance(value2, torch.Tensor)
            value1, value2 = value1.cpu(), value2.cpu()
            if not (value1.shape == value2.shape and torch.allclose(value1, value2)):
                print(f"Mismatch found at {key}.")
                return False
        elif isinstance(value1, dict):
            assert isinstance(value2, dict)
            if not are_dicts_equal(value1, value2, only_use_common_keys=only_use_common_keys):
                return False
        else:
            if (np.isscalar(value1) and value1 != value2) or (not np.isscalar(value1) and np.any(value1 != value2)):
                print(f"Mismatch found at {key}: {value1} != {value2}")
                return False

    return True


def apply_func_to_dataclasses_list(func, dataclass_instances):
    assert len(dataclass_instances) > 0
    for instance in dataclass_instances:
        assert dataclasses.is_dataclass(instance)

    outputs = {}
    for field in dataclasses.fields(dataclass_instances[0]):
        values = []
        for instance in dataclass_instances:
            value = getattr(instance, field.name)
            if value is not None:
                values.append(value)
        if not values:
            values = None
        else:
            values = func(values)
        outputs[field.name] = values
    output = type(dataclass_instances[0])(**outputs)
    return output


def torch_cat_dataclasses_list(dataclass_instances, dim=0):
    return apply_func_to_dataclasses_list(lambda x: torch.cat(x, dim=dim), dataclass_instances)


def torch_stack_dataclasses_list(dataclass_instances, dim=0):
    return apply_func_to_dataclasses_list(lambda x: torch.stack(x, dim=dim), dataclass_instances)


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]
