import numpy as np
from typing import List
from numba import njit

import torch


def obj_to_slot_format(compressed_obj_encoding, is_obj_visible):
    """
    Convert obj encoding, obj visible maks, and obj existence mask to slot format
        If an object disappears, its slot will be occupied by another object appears at a later frame.
    :param compressed_obj_encoding: (T, num_encoder_slots, slot_dim)
    :param is_obj_visible: (T, num_objs)
    :return:
        encoding: (T, num_slots, slot_dim)
        slots_visible: (T, num_slots)
        slots_exist: (T, num_slots)
    """

    slot_dim, slot_dtype = compressed_obj_encoding.shape[-1], compressed_obj_encoding.dtype
    T, num_objs = is_obj_visible.shape

    # convert compressed_obj_encoding from (T, num_encoder_slots, slot_dim) to (T, num_objs, slot_dim)
    obj_encoding = np.zeros((T, num_objs, slot_dim), dtype=slot_dtype)
    for t in range(T):
        idx = 0
        for i in range(num_objs):
            if is_obj_visible[t, i]:
                obj_encoding[t, i] = compressed_obj_encoding[t, idx]
                idx += 1

    # obj exists from the first frame it appears to the last frame it is visible
    obj_exists = np.copy(is_obj_visible)
    obj_needs_slot = np.copy(is_obj_visible)
    for i in range(num_objs):
        first_true = last_true = -1
        for t in range(T):
            if is_obj_visible[t, i]:
                if first_true == -1:
                    first_true = t
                last_true = t
        if first_true != -1:
            obj_exists[first_true:last_true + 1, i] = True
            obj_needs_slot[first_true:last_true + 2, i] = True

    num_slots = np.max(np.sum(obj_needs_slot, axis=1))

    encoding = np.zeros((T, num_slots, slot_dim), dtype=slot_dtype)
    slots_visible = np.zeros((T, num_slots), dtype=bool)
    slots_exist = np.zeros((T, num_slots), dtype=bool)

    # initialize the first frame slot with the first frame. If an object disappears, assign its slot to another object
    slot_available = np.ones(num_slots, dtype=bool)
    obj_slot_id = np.full(num_objs, -1, dtype=int)          # mapping from obj_id to slot_id

    for t in range(T):
        # clean out slots whose objects disappear to make them available for new objects
        for i in range(num_objs):
            if not obj_needs_slot[t, i] and obj_slot_id[i] != -1:
                slot_id = obj_slot_id[i]
                slot_available[slot_id] = True
                obj_slot_id[i] = -1

        # add new objects to the available slots and update existing objects
        for i in range(num_objs):
            if obj_needs_slot[t, i]:
                if obj_slot_id[i] == -1:
                    for j in range(num_slots):
                        if slot_available[j]:
                            slot_available[j] = False
                            obj_slot_id[i] = j
                            break

                slot_id = obj_slot_id[i]
                encoding[t, slot_id] = obj_encoding[t, i]
                slots_visible[t, slot_id] = is_obj_visible[t, i]
                slots_exist[t, slot_id] = obj_exists[t, i]

    return encoding, slots_visible, slots_exist


def concat_tensor_fn(
    batch: List[torch.Tensor],
):
    elem = batch[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        seq_len = sum(x.size(0) for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(seq_len, *list(elem.size()[1:]))
    return torch.concat(batch, 0, out=out)


@njit
def get_temporal_idxes(slots_exist, slots_visible, temporal_random_idxes, slot_obj_id):
    """
    Get a random slot idx from the same object
    :param temporal_random_idxes, slots_exist, slots_visible: (T, num_slots)
    """
    T, num_slots = slots_exist.shape
    for i in range(num_slots):
        idxes = []
        obj_id = 0
        exist_prev_t = False
        for t in range(T):
            if slots_exist[t, i]:
                exist_prev_t = True
                if slots_visible[t, i]:
                    idxes.append(t)
                    slot_obj_id[t, i] = obj_id
            else:
                if exist_prev_t:
                    obj_id += 1
                exist_prev_t = False

                shuffled_idxes = np.random.choice(np.array(idxes), len(idxes), replace=True)
                # shuffled_idxes = idxes[::-1]
                for idx, new_idx in zip(idxes, shuffled_idxes):
                    temporal_random_idxes[idx, i] = new_idx

                # essentially an empty list, but trick to avoid numba type error
                idxes = [np.int64(x) for x in range(0)]

        shuffled_idxes = np.random.choice(np.array(idxes), len(idxes), replace=True)
        # shuffled_idxes = idxes[::-1]
        for idx, new_idx in zip(idxes, shuffled_idxes):
            temporal_random_idxes[idx, i] = new_idx
    return temporal_random_idxes, slot_obj_id
