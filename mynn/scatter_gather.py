import torch
import numpy as np
import torch.cuda.comm as comm

def scatter(inputs, target_gpus, dim=0):
    obj=inputs[0]
    def scatter_map_simple(obj):
        Len=len(obj)
        chunk_size=Len//len(target_gpus)
#        print('Each device contains {} samples'.format(chunk_size))
        return [obj[i*chunk_size:(i+1)*chunk_size] for i in range(len(target_gpus))]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
#        return scatter_map(inputs)
        return scatter_map_simple(obj)
    finally:
        scatter_map_simple = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    Outputs is a list of output of each device
    """
    def gather_map_simple(outputs, target_device):
        output=outputs[0]
        for out in outputs[1:]:
            for k, v in out.items():
                if isinstance(v, list):
                    output[k].extend(v)
                elif isinstance(v, torch.Tensor):
#                    output[k]=torch.cat((output[k], v), dim=0)
                    output[k]=comm.gather((output[k], v), 0, target_device)
                elif isinstance(v, np.ndarray):
                    output[k]=np.append(output[k], v, axis=0)
                elif isinstance(v, int) or isinstance(v, float):
                    output[k]+=v
                else:
                    raise TypeError
        return output

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map_simple(outputs, target_device)
    finally:
        gather_map_simple = None
