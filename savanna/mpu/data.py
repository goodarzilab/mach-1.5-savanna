# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_src_rank
from .initialize import get_sequence_parallel_group
from .initialize import get_sequence_parallel_rank
from .initialize import get_sequence_parallel_src_rank
from .initialize import get_sequence_parallel_world_size


_MAX_DATA_DIM = 4


def _check_data_types(keys, data, target_dtype):
    """Check that all the keys have the same target data type."""
    for key in keys:
        assert data[key].dtype == target_dtype, "{} has data type {} which " "is different than {}".format(
            key, data[key].dtype, target_dtype
        )


def _build_key_size_numel_dictionaries(keys, data):
    """Build the size on rank 0 and broadcast."""
    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]

    # Pack the sizes on rank zero.
    if (get_model_parallel_rank() == 0) and (get_sequence_parallel_rank() == 0):
        offset = 0
        for key in keys:
            assert data[key].dim() < max_dim, "you should increase MAX_DATA_DIM"
            size = data[key].size()
            for i, s in enumerate(size):
                sizes[i + offset] = s
            offset += max_dim

    # Move to GPU and broadcast.
    sizes_cuda = torch.cuda.LongTensor(sizes)
    torch.distributed.broadcast(sizes_cuda, get_model_parallel_src_rank(), group=get_model_parallel_group())
    torch.distributed.broadcast(sizes_cuda, get_sequence_parallel_src_rank(), group=get_sequence_parallel_group())

    # Move back to cpu and unpack.
    sizes_cpu = sizes_cuda.cpu()
    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel


def broadcast_data(keys, data, datatype):
    """Broadcast data from rank zero of each model parallel group to the
    members of the same model parallel group.

    Arguments:
        keys: list of keys in the data dictionary to be broadcasted
        data: data dictionary of string keys and cpu tensor values.
        datatype: torch data type of all tensors in data associated
                  with keys.
    """
    # Build (key, size) and (key, number of elements) dictionaries along
    # with the total number of elements on all ranks.
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(keys, data)

    # Pack on rank zero.
    if (get_model_parallel_rank() == 0) and (get_sequence_parallel_rank() == 0):
        # Check that all keys have the same data type.
        _check_data_types(keys, data, datatype)
        # Flatten the data associated with the keys
        flatten_data = torch.cat([data[key].contiguous().view(-1) for key in keys], dim=0).cuda()
    else:
        flatten_data = torch.empty(total_numel, device=torch.cuda.current_device(), dtype=datatype)

    # Broadcast
    torch.distributed.broadcast(
        flatten_data, get_model_parallel_src_rank(), group=get_model_parallel_group()
    )
    torch.distributed.broadcast(
        flatten_data, get_sequence_parallel_src_rank(), group=get_sequence_parallel_group()
    )

    # Unpack
    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output


def zigzag_split_across_cp_ranks(data, seq_dim=1, verbose=False):
    """Splits the data along the seq dimension in a zigzag fashion.
    Arguments:
        data: data dictionary of string keys and cpu tensor values.
        seq_dim: the sequence dimension to split.
        verbose: whether to print debug info.
    """

    worldsize = get_sequence_parallel_world_size()
    # first check if we can just skip it...
    if worldsize == 1:
        return data

    cp_rank = get_sequence_parallel_rank()

    # Zigzag-split the data
    seq_chunks = torch.chunk(data, 2 * worldsize, dim=seq_dim)
    _data = [
        torch.cat((seq_chunks[i], seq_chunks[-(i + 1)]), dim=seq_dim)
        for i in range(worldsize)
    ]

    if verbose:
        torch.distributed.barrier()
        print(f"[rank={torch.distributed.get_rank()}] | In data_zigzag function.\n")
        print(f"[rank={torch.distributed.get_rank()}] | Initial data shape={data.shape}. Zigzagging data along seq_dim={seq_dim}\n")
        print(f"[rank={torch.distributed.get_rank()}] | Chunk size = {_data[cp_rank].shape}\n")
        torch.distributed.barrier()

    # Select the corresponding rank
    return _data[cp_rank].contiguous()


def zigzag_gather_from_cp_ranks(data, seq_dim=1, verbose=False):
    """
    Gathers data from all context parallel ranks according to zigzag splitting.
    Arguments:
        data: data dictionary of string keys and cpu tensor values.
        seq_dim: the sequence dimension to gather.
        verbose: whether to print debug info.
    """

    worldsize = get_sequence_parallel_world_size()
    # first check if we can just skip it...
    if worldsize == 1:
        return data

    cp_group = get_sequence_parallel_group()
    cp_rank = get_sequence_parallel_rank()

    # Gather from all ranks using autograd-enabled all_gather
    gathered_data = torch.distributed.nn.functional.all_gather(data, group=cp_group)

    # Initialize a list to store the original sequence chunks
    # `gathered_data` is a list of tensors from all ranks
    # Each rank's data consists of two chunks concatenated along seq_dim
    seq_chunks = [None] * (2 * worldsize)

    if verbose:
        torch.distributed.barrier()
        print("On De-zigzag function.")

    for i, data_i in enumerate(gathered_data):
        chunk_size = data_i.size(seq_dim) // 2

        # Split the data_i back into the original two chunks
        chunk0, chunk1 = torch.split(data_i, chunk_size, dim=seq_dim)

        # Reassign the chunks to their original positions
        seq_chunks[i] = chunk0
        seq_chunks[-(i + 1)] = chunk1

        if verbose:
            print(f"[rank={cp_rank}] | Values | [i] = {i} and [-(i + 1)] = {-(i + 1)}\n")
            print(f"[rank={cp_rank}] | Retrieved chunks from rank {i}: chunk0 shape= {chunk0.shape} / chunk1 shape= {chunk1.shape}\n")

        # Concatenate all chunks to reconstruct the original data
    reconstructed_data = torch.cat(seq_chunks, dim=seq_dim)

    if verbose:
        print(f"[rank={cp_rank}] | Reconstructed data shape: {reconstructed_data.shape}\n")
        torch.distributed.barrier()

    return reconstructed_data