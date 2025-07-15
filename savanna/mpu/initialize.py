"""Model and data parallel groups."""
import torch

from .utils import ensure_divisibility

# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Pipeline parallel group that the current rank belongs to.
_PIPE_PARALLEL_GROUP = None
# Context parallel group that the current rank belongs to.
_SEQUENCE_PARALLEL_GROUP = None
# Used for Deepspeed for context parallel
# The term sequence is used here for parity with Deepspeed,
# since the library uses that naming for finding the functions
_SEQUENCE_DATA_PARALLEL_GROUP = None

# A group used to sync during the IO process. Usually this is data_parallel_group(),
# but with pipeline parallelism it must also involve the last stage (which is not in the
# DP group of rank 0)
_IO_PARALLEL_GROUP = None

# A group used to sync FP8 activation stats. This includes all ranks on the same
# pipeline depth, regardless of model or data parallel position.
_FP8_SYNC_GROUP = None

# These values enable us to change the mpu sizes on the fly.
_MPU_WORLD_SIZE = None
_MPU_RANK = None

# Used to query 4D topology
_MPU_TOPOLOGY = None

# Get fp32_allreduce flag
_FP32_ALLREDUCE = None


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def initialize_model_parallel(
    model_parallel_size,
    pipe_parallel_size,
    context_parallel_size,
    data_parallel_size,
    topology=None,
    fp32_allreduce=False
):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used for model parallelism.
        pipe_parallel_size: number of GPUs used for pipeline parallelism.
        context_parallel_size: number of GPUs used for context parallelism.
        topology: topology if it exists.
        fp32_allreduce: whether or not to do all reduce in fp32.
    Adjacent ranks are ordered by model parallel, then context parallel,
    then data parallel.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel groups as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    if torch.distributed.get_rank() == 0:
        print("> initializing model parallel with size {}".format(model_parallel_size))
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    if world_size < model_parallel_size * context_parallel_size:
        raise ValueError(
            "world size cannot be smaller than (model parallel size) * (sequence parallel size)"
        )
    if pipe_parallel_size > 1 and context_parallel_size > 1:
        # This just hasn't been thought through and tested, it might not be that
        # much additional work to support.
        raise ValueError(
            "pipeline parallel not supported with context parallel"
        )
    ensure_divisibility(world_size, model_parallel_size)
    rank = torch.distributed.get_rank()

    global _MPU_TOPOLOGY
    if topology:
        _MPU_TOPOLOGY = topology

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    if topology:
        for dp_group in topology.get_axis_comm_lists("data"):
            group = torch.distributed.new_group(ranks=dp_group)
            if rank == 0:
                print("MPU DP:", dp_group)
            if rank in dp_group:
                _DATA_PARALLEL_GROUP = group
    else:
        for i in range(model_parallel_size * context_parallel_size):
            ranks = range(i, world_size, model_parallel_size * context_parallel_size)
            group = torch.distributed.new_group(ranks)
            if i == (rank % (model_parallel_size * context_parallel_size)):
                _DATA_PARALLEL_GROUP = group

    # Build pipeline parallel group
    if topology is not None:
        global _PIPE_PARALLEL_GROUP
        for pp_group in topology.get_axis_comm_lists("pipe"):
            group = torch.distributed.new_group(ranks=pp_group)
            if rank == 0:
                print("MPU PP:", pp_group)
            if rank in pp_group:
                _PIPE_PARALLEL_GROUP = group

    # Build IO group
    global _IO_PARALLEL_GROUP
    if topology and topology.get_dim("pipe") > 1:
        io_stages = [0, topology.get_dim("pipe") - 1]
        io_group = []
        for stage in io_stages:
            io_group.extend(topology.filter_match(pipe=stage, model=0))
        if rank == 0:
            print("MPU IO:", io_group)
        group = torch.distributed.new_group(ranks=io_group)
        if rank in io_group:
            _IO_PARALLEL_GROUP = group
    else:
        _IO_PARALLEL_GROUP = get_data_parallel_group()
        
    global _SEQUENCE_PARALLEL_GROUP
    assert _SEQUENCE_PARALLEL_GROUP is None, "context parallel group is already initialized"
    if topology:
        for cp_group in topology.get_axis_comm_lists("context"):
            group = torch.distributed.new_group(ranks=cp_group)
            if rank == 0:
                print("MPU CP:", cp_group)
            if rank in cp_group:
                _SEQUENCE_PARALLEL_GROUP = group
    else:
        for i in range(world_size // context_parallel_size):
            ranks = range(i * model_parallel_size * context_parallel_size, (i + 1) * model_parallel_size * context_parallel_size, model_parallel_size)
            group = torch.distributed.new_group(ranks)
            if i in ranks:
                _SEQUENCE_PARALLEL_GROUP = group

    global _FP8_SYNC_GROUP
    if topology and topology.get_dim("pipe") > 1:
        pipe_depth = topology.get_dim("pipe")
        for stage in range(pipe_depth):
            pipe_group = topology.filter_match(pipe=stage)
            if rank == 0:
                print("MPU FP8:", pipe_group)
            group = torch.distributed.new_group(ranks=pipe_group)
            if rank in pipe_group:
                _FP8_SYNC_GROUP = group
    else:
        # If pipe depth is 1, then we sync amongst all ranks
        _FP8_SYNC_GROUP = None

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    if topology:
        for mp_group in topology.get_axis_comm_lists("model"):
            group = torch.distributed.new_group(ranks=mp_group)
            if rank == 0:
                print("MPU MP:", mp_group)
            if rank in mp_group:
                _MODEL_PARALLEL_GROUP = group

    else:
        for i in range(world_size // model_parallel_size):
            ranks = range(i * model_parallel_size, (i + 1) * model_parallel_size)
            group = torch.distributed.new_group(ranks)
            if i == (rank // model_parallel_size):
                _MODEL_PARALLEL_GROUP = group

    global _SEQUENCE_DATA_PARALLEL_GROUP
    assert _SEQUENCE_DATA_PARALLEL_GROUP is None, "sequence data parallel group is already initialized"
    if context_parallel_size > 1:
        if topology:
            sdp_groups = [[] for _ in range(model_parallel_size)]
            for mp_group in topology.get_axis_comm_lists("model"):
                for i, mp_rank in enumerate(mp_group):
                    sdp_groups[i].append(mp_rank)
            for sdp_group in sdp_groups:
                group = torch.distributed.new_group(ranks=sdp_group)
                if rank == 0:
                    print("MPU SDP:", sdp_group)
                if rank in sdp_group:
                    _SEQUENCE_DATA_PARALLEL_GROUP = group
        else:
            sdp_groups = [[]] * model_parallel_size
            for i in range(world_size // model_parallel_size):
                sdp_groups[i % model_parallel_size].append(i)
            for sdp_group in sdp_groups:
                group = torch.distributed.new_group(ranks=sdp_group)
                if rank in sdp_group:
                    _SEQUENCE_DATA_PARALLEL_GROUP = group
    else:
        _SEQUENCE_DATA_PARALLEL_GROUP = _DATA_PARALLEL_GROUP

    global _FP32_ALLREDUCE
    assert _FP32_ALLREDUCE is None, "fp32_allreduce is already initialized"
    _FP32_ALLREDUCE = fp32_allreduce


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group(check_initialized=True):
    """Get the model parallel group the caller rank belongs to."""
    if check_initialized:
        assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP

get_tensor_model_parallel_group = get_model_parallel_group

def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_io_parallel_group():
    """Get the IO parallel group the caller rank belongs to."""
    assert _IO_PARALLEL_GROUP is not None, "IO parallel group is not initialized"
    return _IO_PARALLEL_GROUP


def set_model_parallel_world_size(world_size):
    """Set the model parallel size"""
    global _MPU_WORLD_SIZE
    _MPU_WORLD_SIZE = world_size


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    global _MPU_WORLD_SIZE
    if _MPU_WORLD_SIZE is not None:
        return _MPU_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_model_parallel_group())

get_tensor_model_parallel_world_size = get_model_parallel_world_size

def set_model_parallel_rank(rank):
    """Set model parallel rank."""
    global _MPU_RANK
    _MPU_RANK = rank


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    global _MPU_RANK
    if _MPU_RANK is not None:
        return _MPU_RANK
    return torch.distributed.get_rank(group=get_model_parallel_group())

get_tensor_model_parallel_rank = get_model_parallel_rank

def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_sequence_parallel_group():
    """Get the context parallel group the caller rank belongs to."""
    assert (
        _SEQUENCE_PARALLEL_GROUP is not None
    ), "context parallel group is not initialized"
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_data_parallel_group():
    """Get the sequence data parallel group the caller rank belongs to."""
    assert _SEQUENCE_DATA_PARALLEL_GROUP is not None, "sequence data parallel group is not initialized"
    return _SEQUENCE_DATA_PARALLEL_GROUP


def get_sequence_parallel_world_size():
    """Return world size for the context parallel group."""
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())


def get_sequence_parallel_rank():
    """Return my rank for the context parallel group."""
    return torch.distributed.get_rank(group=get_sequence_parallel_group())


def get_sequence_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the sequence parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = (
        get_model_parallel_world_size() * get_sequence_parallel_world_size()
    )
    return (
        global_rank // local_world_size
    ) * local_world_size + get_model_parallel_rank()


def get_sequence_data_parallel_rank():
    """Return my rank for the sequence data parallel group."""
    return torch.distributed.get_rank(group=get_sequence_data_parallel_group())


def get_sequence_data_parallel_world_size():
    """Return world size for sequence data parallel group."""
    return torch.distributed.get_world_size(group=get_sequence_data_parallel_group())


def get_data_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zero
    in the data parallel group."""
    global_rank = torch.distributed.get_rank()
    topo = get_topology()
    if topo is None:
        # we are just using model parallel
        return global_rank % get_model_parallel_world_size() % get_sequence_parallel_world_size()
    else:
        # We are using pipeline parallel
        d = topo.get_axis_comm_lists("data")
        for l in d:
            if global_rank in l:
                return l[0]


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def get_topology():
    return _MPU_TOPOLOGY


def get_pipe_parallel_group():
    """Get the pipe parallel group the caller rank belongs to."""
    assert _PIPE_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _PIPE_PARALLEL_GROUP


def get_pipe_parallel_rank():
    """Return my rank for the pipe parallel group."""
    return torch.distributed.get_rank(group=get_pipe_parallel_group())


def get_pipe_parallel_world_size():
    """Return world size for the pipe parallel group."""
    return torch.distributed.get_world_size(group=get_pipe_parallel_group())


def get_fp8_sync_group():
    """Get the pipe parallel group the caller rank belongs to."""
    return _FP8_SYNC_GROUP


def get_fp8_sync_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_fp8_sync_group())


def get_fp8_sync_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_fp8_sync_group())


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _PIPE_PARALLEL_GROUP
    _PIPE_PARALLEL_GROUP = None
    global _IO_PARALLEL_GROUP
    _IO_PARALLEL_GROUP = None
    global _FP8_SYNC_GROUP
    _FP8_SYNC_GROUP = None
    global _MPU_WORLD_SIZE
    global _MPU_RANK
    _MPU_WORLD_SIZE = None
    _MPU_RANK = None
    global _MPU_TOPOLOGY
    _MPU_TOPOLOGY = None
    global _FP32_ALLREDUCE
    _FP32_ALLREDUCE = None
    global _SEQUENCE_PARALLEL_GROUP
    _SEQUENCE_PARALLEL_GROUP = None
    global _SEQUENCE_DATA_PARALLEL_GROUP
    _SEQUENCE_DATA_PARALLEL_GROUP = None


def get_fp32_allreduce():
    """Get the fp32 allreduce flag"""
    assert _FP32_ALLREDUCE is not None, "fp32_allreduce is not Initialized"
    return _FP32_ALLREDUCE
