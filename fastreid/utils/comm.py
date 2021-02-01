"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
参考链接：https://pytorch.org/docs/stable/distributed.html#collective-functions
"""

import functools
import logging
import numpy as np
import pickle
import torch
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None     # 在engine.launch.py Line110 有设置
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()  # This collective blocks processes until the whole group enters this function


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    基本原则:
        Use the NCCL backend for distributed GPU training
        Use the Gloo backend for distributed CPU training
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    # pickle.dumps序列化对象; 另外dump是将对象序列化并存储到文件
    # 假如data=10, 则buffer形如: b'\x80\x03K\x08.', storage形如128 3 75 8 46 (屏幕上是换行而非这里的空格)
    # tensor形如 tensor([128,3,75,8,46], dtype=torch.uint8), tensor.numel()为5
    # tensor.cpu().numpy().tobytes()可恢复出序列化结果，然后送入pickle.loads()即可恢复出10
    buffer = pickle.dumps(data)   

    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert ( world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [ torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size) ]
    
    # 各GPU分别说出自己的local_size，all_gather后各GPU都能获取到size_list
    dist.all_gather(size_list, local_size, group=group)

    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.   为何默认在CPU上操作? --> evaluation等操作在cpu上进行
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [ torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list ]

    # torch.distributed.all_gather(tensor_list, tensor, group=<object object>, async_op=False)
    #   Gathers tensors from the whole group in a list. 即所有GPU都获取gather的结果
    #   tensor_list (list[Tensor]) – Output list. 
    #       It should contain correctly-sized tensors to be used for output of the collective.
    #   tensor (Tensor) – Tensor to be broadcast from current process.
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.    为何默认在CPU上操作? --> evaluation等操作在cpu上进行
    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [ torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list ]
        
        # torch.distributed.gather(tensor, gather_list=None, dst=0, group=<object object>, async_op=False)
        # Gathers a list of tensors in a single process 即只有目标GPU获取gather结果
        #   tensor (Tensor) – Input tensor.
        #   gather_list (list[Tensor], optional) – List of appropriately-sized tensors to use for gathered data 
        #                                         (default is None, must be specified on the destination rank)
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.
    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum
    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        
        # torch.distributed.reduce(tensor, dst, op=ReduceOp.SUM, group=<object object>, async_op=False)
        #   Reduces the tensor data across all machines. Only the process with rank dst is going to receive the final result.
        #   tensor (Tensor) – Input and output of the collective. The function operates in-place.
        #   dst (int) – Destination rank
        #   op (optional) – One of the values from torch.distributed.ReduceOp enum. 
        #                   Specifies an operation used for element-wise reductions.
        dist.reduce(values, dst=0)

        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case   
            # 默认reduce是求和，所以这里简单除以GPU个数得均值
            values /= world_size        
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
