# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

import subprocess
from dataclasses import dataclass

try:
    from .template import GlobalConfigTemplate
except ImportError:
    from template import GlobalConfigTemplate

try:
    from typing import List, Literal, Union
except ImportError:
    from typing_extensions import List, Literal, Union


operator_type_CHOICES = [
    "global",
    "local",
    "sparse_fixed",
    "sparse_variable",
    "bigbird",
    "bslongformer",
    "gmlp",
    "amlp",
    "flash",
    "flash_v2",
    "hyena",
    "hyena_v2",
    "hyena_jumpy",
    "hyena_outer",
    "hyena_doublegate",
    "hyena_doublegate_v2",
    "hyena_se",
    "hyena_mr",
    "hyena_downsample_2",
    "hyena_downsample_4",
    "hyena_downsample_8",
    "hyena_downsample_16",
    "hyena_downsample_32",
    "hyena_downsample_64",
    "hyena_downsample_128",
    "ring",
    "flash_te",
]


def get_git_commit_hash():
    """Gets the git commit hash of your current repo (if it exists)"""
    try:
        git_hash = subprocess.check_output(["git", "describe", "--always"], stderr=subprocess.DEVNULL).strip()
        git_hash = git_hash.decode()
    except subprocess.CalledProcessError:
        git_hash = None
    return git_hash


@dataclass
class GlobalConfigParallelism(GlobalConfigTemplate):

    pretraining_strategy: str = "AR"
    """
    Pretraining method to use.
    AR = autoregressive
    MLM = masked language modeling
    SPAN = span masking
    SPAN_R = span masking with discrete noise
    OADM = order-agnostic autoregression diffusion modeling
    """

    """
    Parallelism Arguments
    """

    pipe_parallel_size: int = 0
    """
    Number of pipeline parallel stages. Disable with 0.
    """

    model_parallel_size: int = 1
    """
    Size of the model parallelism.
    """
    
    context_parallel_size: int = 1
    """
    Size of the context parallelism.
    """

    use_cp_hyena: bool = False
    """
    Use context parallelism for Hyena.
    """

    use_cp_ring: bool = False
    """
    Use ring attention when using context parallelism.
    """

    use_cp_flash_te: bool = False
    """
    Use transformer engine attention when using context parallelism.
    """

    flash_te_comm_type: Literal["p2p", "all_gather", "a2a"] = "p2p"
    """
    Communication type for transformer engine attention when using context parallelism.
    One of "p2p", "all_gather", "a2a".
    """
    
    te_attn_backend: Literal["FLASH", "FUSED", "UNFUSED"] = "FLASH"
    """
    Backend to use for transformer engine attention.
    """
    
    nvte_debug: bool = False
    """
    Whether to enable transformer engine debug mode.
    """
    
    nvte_debug_level: int = 2
    """
    Debug level for transformer engine: 0, 1, 2.
    """
    
    pipe_partition_method: str = "type:ParallelBlockPipe"
    """
    method used to distribute model layers across pipeline stages. Choose from "parameters", which balances the number
    of parameters on each pipeline stage, "uniform", which naively balances the number of layers per stage, or
    "type:[regex]", which balances layers whose class names match [regex]
    """

    world_size: int = None
    """
    Total world size (i.e number of gpus in cluster). Configured post-launch using distributed launcher
    """

    is_pipe_parallel: bool = False
    """
    flag to determine whether pipeline parallelism is on - shouldn't be set by user, is automatically determined
    according to pipeline parallel size.
    """

    sequence_parallel: bool = False
    """
    flag to determine whether Megatron-style Sequence Parallelism (https://arxiv.org/abs/2205.05198)
    (Layernorm inputs and activations are sharded across model parallel group) will be used. Has no effect when model_parallel_size is 1.
    **Set by user, in contrast to neox_args.is_pipe_parallel.**
    """
    
    is_context_parallel: bool = False
    """
    flag to determine whether context parallelism is on - shouldn't be set by user, is automatically determined
    according to context parallel size.
    """


@dataclass
class GlobalConfigModel(GlobalConfigTemplate):
    """
    Model Arguments
    """

    tie_projection_weights: bool = False
    """
    Tie projection weights between QKV for attn and hyena (will repeat output 3 times).
    """

    to_upper: str = None
    """
    "upper"
    "weighted"
    Whether to convert all text to uppercase.
    """

    lowercase_loss_reweighting: float = 0.1
    """
    If to_upper == "weighted" or "normalized_weighted"
    Weight to apply to lowercase tokens in the loss function, 1.0 is no reweighting.
    """

    mask_loss_control_tags: bool = False
    """
    Mask loss on control tags:
    '|d__Eukaryota;p__Chordata;c__Mammalia;o__Primates;f__Hominidae;g__Homo;s__Homo sapiens|' phylogenetic tags
    '@' for splice overhang dataset
    """

    precision: Literal["fp16", "fp32", "bfloat16", "fp8"] = None
    """
    description of the used precision. If fp8, will use the Transformer Engine fp8 context manager
    in all layers that support it.
    """

    use_fp8_linears: bool = False
    """
    Deprecated, use use_fp8_mlp_projections instead. If turned on, will only activate fp8 on mlp projections.
    """

    use_fp8_mlp_projections: bool = False
    """
    Activates Transformer Engine FP8 linears for MLPs.
    """

    disable_fp8_w3: bool = False
    """
    Disable fp8 w3 in FFN, needed when weights don't meet TE shape requirements (shape[0] % 8 == 0 and shape[1] % 16 == 0).
    """

    pad_mlp_weights: bool = False
    """
    Pad MLP weights to meet TE shape requirements (shape[0] % 8 == 0 and shape[1] % 16 == 0).
    """

    use_fp8_hyena_mlp_input_projections: bool = False
    """
    Activates Transformer Engine FP8 linears for Hyena MLP input projections.
    """

    use_fp8_hyena_mlp_output_projections: bool = False
    """
    Activates Transformer Engine FP8 linears for Hyena MLP output projections.
    """

    use_fp8_input_projections: bool = False
    """
    Activates Transformer Engine FP8 linears for input projections within mixer layers.
    """

    use_fp8_output_projections: bool = False
    """
    Activates Transformer Engine FP8 linears for output projections within mixer layers.
    """

    num_layers: int = None
    """
    Number of transformer layers.
    """

    hidden_size: int = None
    """
    Transformer hidden size.
    """

    num_attention_heads: int = None
    """
    Number of transformer attention heads.
    """

    seq_length: int = None
    """
    Maximum sequence length to process.
    """

    seq_dim: int = 1
    """
    Dimension of the sequence length of inputs to ParallelSequenceMixer layers.  [B L D] -> seq_dim = 1.
    """
    
    permute_glu: bool = False
    """
    Whether to permute inputs and outputs to ParallelGLU when using sequence_parallel
    """
    
    interleave_projections: bool = True
    """
    Whether to interleave input projections before splitting across parallel groups and passing to mixer.

    This should almost always be set to True.  The only case where it should NOT be True is if for some reason you are continuing to pre-train
    a legacy evo2 checkpoint trained without interleaving at the same MP size.
    """

    max_position_embeddings: int = None
    """
    Maximum number of position embeddings to use. This is the size of position embedding.
    """

    gradient_accumulation_fusion: bool = False
    """
    Not implemented
    """

    use_fp8_norm: bool = False
    """
    User transfer_engine for fp8.
    """

    norm: Literal["layernorm", "rmsnorm", "scalenorm"] = "layernorm"
    """
    Normalization layer to use. Choose from "layernorm", "rmsnorm", "scalenorm".
    """

    prenorm: bool = True
    """
    Apply normalization before the mixer.
    """

    postnorm: bool = False
    """
    Apply normalization after the mixer.
    """

    pre_mlp_norm: bool = False
    """
    Apply normalization before the MLP.
    """

    outer_mlp_norm: bool = False
    """
    Apply normalization after the MLP.
    """

    safe: bool = False

    layernorm_epsilon: float = 1.0e-5
    """
    Layer norm epsilon.
    """

    rms_norm_epsilon: float = 1.0e-8
    """
    Root mean squared norm epsilon
    """

    scalenorm_epsilon: float = 1.0e-8
    """
    Scalenorm epsilon
    """

    pos_emb: Literal[
        "learned",
        "rotary",
        "sinusoidal",
        "rpe",
        "alibi",
        "none",
        "rotary2",
        "rotary_linear_scaled",
        "rotary_ntk_scaled",
    ] = "learned"
    """
    Type of positional embedding to use - choose from 'learned', 'rotary', 'sinusoidal', 'rpe', 'none', 'rotary2', 'rotary_linear_scaled', 'rotary_ntk_scaled'
    """

    rpe_num_buckets: int = 32
    """
    T5 relative positional encoding number of buckets, default 32.
    """

    rpe_max_distance: int = 128
    """
    T5 relative positional encoding max distance, default 128.
    """

    opt_pos_emb_offset: int = 0
    """
    Learned position embedding offset (only used by OPT, where it should be set to 2).
    """

    no_weight_tying: bool = False
    """
    Disables weight tying between embedding weights and final Linear layer
    """

    materialize_attn_mask: bool = False
    """
    Materialize the attention mask manually. If using flash attention v2, this should be set to False.
    """

    operator_config: list = None

    """
    operator list for the backbone

    The first item in the list specifies the operator type(s), and should be a list of strings. The second item
    specifies the number of times to repeat those operator types in the full list.
    """

    use_flashfft: bool = False
    """
    Use flashfftconv instead of torch fft kernel (requires installation of flashfftconv)for hyena
    """

    use_cgcg: bool = False
    """
    Use cgcg (chunked gate-conv-gate) kernel for hyena
    """

    use_cgcg_short: bool = False
    """
    Use cgcg (chunked gate-conv-gate) kernel for hyena short conv
    """

    use_cgcg_mlp: bool = False
    """
    Use cgcg (chunked gate-conv-gate) kernel for hyena mlp
    """

    cgcg_dtype: str = "float32"
    """
    dtype to use within cgcg kernel
    """

    cgcg_fwd_autotune: bool = False
    """
    Whether to autotune cgcg fwd kernel

    @jeromeku: Note autotuning fwd kernel is unstable,
    use pre-tuned config for now.
    """

    cgcg_medium_fwd_kernel_config_chunk_size: int = 128
    """
    cgcg fwd medium conv kernel config chunk size
    """
    cgcg_medium_fwd_kernel_config_block_d: int = 128
    """
    cgcg fwd medium conv kernel config block d tile size
    """

    cgcg_medium_fwd_kernel_config_threadblock_swizzle: str = "row"
    """
    cgcg fwd medium conv kernel config threadblock swizzle type
    """
    cgcg_medium_fwd_kernel_config_chunk_tiles_per_program: int = 3
    """
    cgcg fwd medium conv kernel config chunk tiles per program
    """

    cgcg_medium_fwd_kernel_config_num_warps: int = 4
    """
    cgcg fwd short conv kernel config num warps
    """

    cgcg_medium_fwd_kernel_config_num_stages: int = 3
    """
    cgcg fwd medium conv kernel config num mma pipeline stages
    """

    cgcg_short_fwd_kernel_config_chunk_size: int = 128
    """
    cgcg fwd short conv kernel config chunk size
    """
    cgcg_short_fwd_kernel_config_block_d: int = 128
    """
    cgcg fwd short conv kernel config block d tile size
    """

    cgcg_short_fwd_kernel_config_threadblock_swizzle: str = "row"
    """
    cgcg fwd short conv kernel config threadblock swizzle type
    """
    cgcg_short_fwd_kernel_config_chunk_tiles_per_program: int = 1
    """
    cgcg fwd short conv kernel config chunk tiles per program
    """

    cgcg_short_fwd_kernel_config_num_warps: int = 4
    """
    cgcg fwd short conv kernel config num warps
    """

    cgcg_short_fwd_kernel_config_num_stages: int = 1
    """
    cgcg fwd short conv kernel config num mma pipeline stages
    """

    cgcg_bwd_autotune: bool = True
    """
    Whether to autotune cgcg bwd kernel
    """

    cgcg_fused_bwd: bool = True
    """
    Whether to use fused cgcg bwd kernel
    """

    ## TODO: jeromeku Need two pairs of bwd kernel configs -- pre/post for medium, pre/post for short

    cgcg_bwd_kernel_config_pre_conv_block_x: int = 128
    """
    cgcg bwd pre_conv kernel config block x tile size
    """

    cgcg_bwd_kernel_config_pre_conv_block_y: int = 128
    """
    cgcg bwd pre_conv kernel config block y tile size
    """

    cgcg_bwd_kernel_config_pre_conv_num_warps: int = 8
    """
    cgcg bwd pre_conv kernel config num warps
    """

    cgcg_bwd_kernel_config_pre_conv_num_stages: int = 2
    """
    cgcg bwd pre_conv kernel config num warps
    """

    cgcg_bwd_kernel_config_post_conv_block_x: int = 32
    """
    cgcg bwd post conv kernel config block x tile size
    """

    cgcg_bwd_kernel_config_post_conv_block_y: int = 128
    """
    cgcg bwd post conv kernel config block y tile size
    """

    cgcg_bwd_kernel_config_post_conv_num_warps: int = 4
    """
    cgcg bwd post conv kernel config num warps
    """

    cgcg_bwd_kernel_config_post_conv_num_stages: int = 2
    """
    cgcg bwd post conv kernel config num stages
    """

    short_conv_L: int = None
    """
    For Hyena models, length of the short convolution.
    """

    use_hyena_filter: bool = False
    """
    Whether to use the Hyena filter.
    """

    normalize_hyena_filters: bool = False

    conv_proj_bias: bool = True
    """
    Use bias in the short conv1D, needed for model parallel for the short conv.
    """

    use_fast_heads: bool = False
    """
    Use external fast heads in Hyena mixer (reduce BEFORE fftconv)
    """

    use_slow_heads: bool = False
    """
    Use external outer-product heads in Hyena.
    """

    use_long_conv1d: bool = False

    num_heads: int = 8
    """
    Deprecated, use num_groups_hyena instead.
    """

    num_groups_hyena: int = None
    """
    Determines number of unique filters to have, for the hyena long filter.
    """

    num_groups_hyena_medium: int = None
    """
    Determines number of unique filters to have, for the hyena medium filter.
    """

    num_groups_hyena_short: int = None
    """
    Determines number of unique filters to have, for the hyena short filter.
    """

    num_groups_hyena_mlp: int = None
    """
    Determines number of unique filters to have, for the hyena mlp (filter).
    """

    use_depthwise_short_conv_grouping: bool = True
    """
    Whether to use depthwise convolution grouping for short conv and hyena mlp filters.

    The new filter grouping implements a depthwise (channelwise) convolution whereas the previous version
    implemented a grouped convolution.
    """

    num_heads_2: int = 8
    """
    Number of heads in second type of Hyena.
    """

    hyena_filter_cls: str = "implicit"
    """
    """

    hyena_width_expansion: float = 1.0
    """
    Factor to expand the projections width within hyena layers.
    """
    hyena_medium_filter_cls: str = None
    """
    For medium hyena filters specifically, None defaults ot same as hyena_filter_cls (long filters).
    """

    hyena_filter_r_max: float = 0.99

    hyena_filter_r_min: float = 0.5

    hyena_filter_emb_dim: int = 33

    hyena_filter_fast_decay: float = 0.3

    hyena_filter_slow_decay: float = 0.9

    hyena_filter_order: int = 64

    hyena_filter_num_inner_mlps: int = 2

    hyena_filter_w: int = 14

    hyena_filter_wd: float = 0.0

    hyena_filter_omega_0: float = 1

    hyena_pos_emb: str = "fourier_fixed"

    explicit_filter_decay_preset: str = "normal"

    explicit_filter_num_decay_repeats: int = 1
    
    modal_residue_factors: int = 3

    modal_pole_factors: int = 3

    modal_gamma_min: float = 0.01

    modal_gamma_max: float = 0.1

    use_custom_hyena_short_kernel: bool = False
    """
    Use a custom causal conv layer for the hyena short conv layer.
    """

    use_custom_hyena_mlp_kernel: bool = False
    """
    Use a custom causal conv layer for the hyena short conv layer.
    """

    bidirectional: bool = False
    """
    A bidirectional version of hyena fftconv
    """

    hyena_se_len: int = 3
    """
    Length of the hyena short conv layer, if using
    """

    fast_conv_proj: bool = False
    """
    Use a custom causal conv layer for the hyena projection convs.
    """

    hyena_mr_len: int = 64
    """
    Length of the medium hyena filter.
    """

    fast_conv_mixer: bool = False
    """
    Use a custom causal conv layer for the hyena short conv layer.
    """

    hyena_mlp_len: int = None
    """
    Length of filter used inside the hyena mlp layer. Defaults to hyena_se_len if not provided.
    """

    fast_hyena_mlp_conv: bool = False
    """
    Use a custom causal conv layer for the hyena MLP layer.
    """

    hyena_mlp_expansion_factor: float = 1.0
    """
    Factor to expand the projections width within hyena MLP layers only.
    """

    hyena_mlp_pregate: bool = True
    """
    Use a pre-gate in the hyena MLP layer.
    """

    hyena_mlp_postgate: bool = True
    """
    Use a post-gate in the hyena MLP layer.
    """

    hyena_se_pregate: bool = True
    """
    Use a pre-gate in the hyena short conv layer.
    """

    hyena_se_postgate: bool = True
    """
    Use a post-gate in the hyena short conv layer.
    """

    gating_act: str = "identity"
    """
    Gating to use in Hyena gates
    """

    proj_groups: int = 1

    grouped_attention: bool = False

    log_attn_norms: bool = False

    log_hyena_norms: bool = False

    sparsity_config: dict = None

    """
    Sparsity configuration dict as defined in https://www.deepspeed.ai/docs/config-json/#sparse-attention

    Note that since neox is autoregressive, attention is always "unidirectional" and `horizontal_global_attention` is
    always false.

    The main difference between our sparsity config and deepspeed's is that `mode` is ignored - since it is instead
    specified in operator_config defining each layer.

    An example config is given below:
          "sparse_attention": {
            "block": 16,
            "different_layout_per_head": true,
            "num_local_blocks": 4,
            "num_global_blocks": 1,
            "num_different_global_patterns": 4,
            "num_random_blocks": 0,
            "local_window_blocks": [4],
            "global_block_indices": [0],
            "global_block_end_indices": None,
            "num_sliding_window_blocks": 3
          }
    """

    num_unique_layers: int = None
    """
    Number of unique transformer layers. num-layers should be divisible by this value. Currently only has an effect when pipe_parallel_size=0.
    """

    param_sharing_style: str = "grouped"
    """
    Ordering of the shared parameters. For example, for a num-layers=4 and --num-unique-layers=2, we will have the following ordering for two unique layers 1 and 2-: grouped: [1, 2, 1, 2] and spaced: [1, 1, 2, 2].
    """

    make_vocab_size_divisible_by: int = 128
    """
    Pad the vocab size to be divisible by this value. This is added for computational efficiency reasons.
    """

    activation: Literal["gelu", "geglu", "relu", "softsign", "swish", "mish", "silu", "identity"] = "gelu"
    """
    Activation function to use - choose from ["gelu", "geglu", "relu", "softsign", "swish", "mish"]
    """

    parallel_glu_activation_default: str = None
    """
    """

    scaled_upper_triang_masked_softmax_fusion: bool = False
    """
    Enable fusion of dense_projection_scaling time (upper diagonal) masking and softmax.
    """

    scaled_masked_softmax_fusion: bool = False
    """
    Enable fusion of dense_projection_scaling general masking and softmax.
    """

    bias_gelu_fusion: bool = False
    """
    Enable bias and gelu fusion.
    """

    bias_dropout_fusion: bool = False
    """
    Enable bias and dropout fusion.
    """

    fp16_lm_cross_entropy: bool = False
    """
    Move the cross entropy unreduced loss calculation for lm head to fp16.
    """

    init_method_std: float = 0.02
    """
    Standard deviation of the zero mean normal distribution used for weight initialization.
    """

    apply_query_key_layer_scaling: bool = False
    """
    Scale Q * K^T by 1 / layer-number. If this flag is set, then it will automatically set attention-softmax-in-fp32 to true
    """

    use_cpu_initialization: bool = False
    """
    If set, affine parallel weights initialization uses CPU
    """

    attention_softmax_in_fp32: bool = False
    """
    Run attention masking and softmax in fp32.
    """

    rotary_pct: float = 1.0
    """
    pct of hidden dims to apply rotary positional embedding to
    """

    rotary_emb_base: int = 10000
    """
    Base for rotary positional embedding
    """

    rotary_emb_scaling_factor: float = 1.0
    """
    Scale factor for new sequence length, 1.0 for no scaling, 2.0 means double
    """

    rotary_emb_max_unscaled_seq_len: int = 8192
    """
    For finetuning, we need to know the previous max seq len of the pretrained model
    """

    init_method: Literal[
        "normal",
        "scaled_normal",
        "orthogonal",
        "scaled_orthogonal",
        "xavier_uniform",
        "xavier_normal",
        "wang_init",
        "small_init",
    ] = "normal"
    """
    Init function used on all layers except ff residual outputs - choose from
    ["normal", "scaled_normal", "orthogonal", "scaled_orthogonal", "xavier_uniform", "xavier_normal", "wang_init", "small_init"]
    """

    output_layer_init_method: Literal[
        "normal",
        "scaled_normal",
        "orthogonal",
        "scaled_orthogonal",
        "xavier_uniform",
        "xavier_normal",
        "wang_init",
        "small_init",
    ] = "scaled_normal"
    """
    Init function used for ff residual outputs - choose from
    ["normal", "scaled_normal", "orthogonal", "scaled_orthogonal", "xavier_uniform", "xavier_normal", "wang_init", "small_init"]
    """

    gmlp_attn_dim: int = 64
    """
    the dimension of the single head self attention in gmlp model (not used in gpt models).
    If None - gmlp model doesn't use attention.
    """

    gpt_j_residual: bool = False
    """
    If false, we use the conventional residual path:
      x = x + attn(ln1(x))
      x = x + mlp(ln2(x))
    Otherwise, we use the residual path from GPT-J, which offers a slight speedup:
      x = ln(x)
      x = x + attn(x) + mlp(x)
    """

    gpt_j_tied: bool = False
    """
    If false, we use
      x = x + attn(ln1(x)) + mlp(ln2(x))
    Otherwise, we tie the layer norms
      y = ln(x)
      x = x + attn(y) + mlp(y)
    """

    soft_prompt_tuning: dict = None
    """
    Dictionary configuring the soft prompt tuning parameters.
    If enabled, will train *only* the soft prompt, and freezes the rest of the model.
    parameters in the dict are:
        'enabled': bool = True # enables soft prompting
        'num_tokens': int = 10 # length of the soft prompt in tokens
        'init_string': str = '' # if provided, initialize the soft prompt with the word embeddings of this string
        'init_range': float = 0.5 # if no init string is provided, initialize the soft prompt with a uniform distribution between -init_range and init_rang
    """

    output_layer_parallelism: Literal["row", "column"] = "row"

    """
    Parameter controlling whether the output layer is parallelized over the hidden dim (row) or the vocab dim (column)
    """

    mlp_type: str = "regular"
    """
    Types:
        regular: Megatron implementation
        llama: LLaMA MLP (SiLU-gated MLP)
        identity
    """

    make_gated_mlp_multiple_of: int = 16
    """
    Set the ff_dim to be a multiple of this value for llama mlp. Useful for sharding / using model parallel properly.
    """

    identity_mlp: bool = False
    """
    If true, replaces the post-mixer (or attention) MLP with an identity function.
    """

    _sequence_type: str = "test"
    "Dummy arg for determined API"

    _mapping = str = "test"
    "Dummy arg for determined API"


@dataclass
class GlobalConfigOptimizer(GlobalConfigTemplate):
    """
    Optimizer Arguments
    """

    optimizer_type: Literal[
        "adam",
        "onebitadam",
        "cpu_adam",
        "cpu_torch_adam",
        "sm3",
        "sophia",
        "madgrad_wd",
        "sgd",
    ] = "adam"
    """
    Type of optimizer to use. Choose from ['adam', 'onebitadam', 'cpu_adam', 'cpu_torch_adam', 'sm3', 'sophia', 'madgrad_wd', 'sgd']
    NOTE: sgd will use MuSGD from Mup. Mup must be enabled for this optimizer.
    """

    use_bnb_optimizer: bool = False
    """
    Whether to enable the bitsandbytes optimizers
    """

    zero_stage: Union[int, List[int], Literal["all"]] = None
    """
    Zero Optimizer stage
    """

    zero_reduce_scatter: bool = None
    """
    Zero: Uses reduce or reduce scatter instead of allreduce to average gradients
    """

    zero_contiguous_gradients: bool = None
    """
    Zero: Copies the gradients to a contiguous buffer as they are produced. Avoids memory fragmentation during backward pass. Only useful when running very large models.
    """

    zero_reduce_bucket_size: int = None
    """
    Zero: Number of elements reduced/allreduced at a time. Limits the memory required for the allgather for large model sizes
    """

    zero_allgather_bucket_size: int = None
    """
    Zero: Number of elements allgathered at a time. Limits the memory required for the allgather for large model sizes
    """

    expandable_segments: bool = False
    """
    Check `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` has been set
    This has to be done externally when launching the job.

    """

    recycle_events: bool = False
    """
    Recycle cuda.Events to prevent dynamic allocations

    Helpful when training at 1k+ GPUs to reduce non-determinism across ranks
    """

    disable_gc: bool = False
    """
    Disable automatic python garbage collection

    Helpful when training at 1k+ GPUs to reduce non-determinism across ranks
    """

    gc_collect_generation: int = 2
    """
    https://docs.python.org/3/library/gc.html#gc.collect

    gc collection generation: 0, 1, or 2
    """

    prealloc_mem: bool = False
    """
    Preallocate memory in lieu of torch CCA

    Helpful when training at 1k+ GPUs to reduce non-determinism across ranks and reduce thrash
    """

    zero_use_leaf_modules: bool = False
    """
    Enable leaf modules when using Zero-3

    zero_leaf_modules must be specified when this flag is enabled.
    """

    zero_leaf_modules: list = None
    """
    Modules to mark as leaf modules, only for Zero-stage 3
    Must be list of strings that can be be used as such getattr(module, leaf_module)
    where module is one of savanna.model.block or savanna.model.operators.hyena.hyena.

    This controls the granularity of zero-3 parameter partitioning.  I.e., if ParallelSequenceMixer is
    set as a leaf module, then the entire ParallelSequenceMixer will be gathered / partitioned as a single unit.

    block modules: 'ParallelSequenceMixer', 'ParallelGLU', 'ParallelLinear', 'FlexLinear', 'ParallelMLP',
    hyena_modules: 'ParallelCausalDepthwiseConv1d', 'ParallelComplexModalFilter', 'ParallelHyenaOperator', 'ParallelImplicitFreeformFilter', 'ParallelShortHyenaOperator',
    """

    zero_use_mics: bool = False
    """
    DEPRECATED

    Use MiCS partitioning for Zero-3
    """

    patch_record_stream: bool = False
    """
    DEPRECATED

    Apply @gdb record_stream patch for torch record_stream
    """

    record_stream_flush_frequency: int = 50
    """
    DEPRECATED

    Frequency to flush tensors and events
    """

    record_stream_backlog: int = 2
    """
    DEPRECATED

    Size of event backlog
    """

    lr: float = None
    """
    Max Learning rate during training
    """
    lr_medium_hyena: float = None
    """
    optional learning rate for the medium hyena filters
    """

    wd_free_lr: float = None
    """
    Max learning rate of weight decay free parameters. If None, defaults to --lr
    """


@dataclass
class GlobalConfigLRScheduler(GlobalConfigTemplate):
    """
    LR Scheduler Arguments
    """

    lr_decay_style: Literal["constant", "linear", "cosine", "exponential"] = "linear"
    """
    Learning rate decay function. Choose from 'constant', 'linear', 'cosine', 'exponential'.
    """

    lr_decay_iters: int = None
    """
    Number of iterations to decay learning rate over, If None defaults to --train-iters
    """

    min_lr: float = 0.0
    """
    Minimum value for learning rate. The scheduler clips values below this threshold.
    """

    warmup: float = 0.01
    """
    Percentage of total iterations to warmup on (.01 = 1 percent of all training iters).
    """

    override_lr_scheduler: bool = False
    """
    Reset the values of the scheduler (learning rate,warmup iterations, minimum learning rate, maximum number of iterations, and decay style from input arguments and ignore values from checkpoints. Note that all the above values will be reset.
    """

    use_checkpoint_lr_scheduler: bool = False
    """
    Use checkpoint to set the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style from checkpoint and ignore input arguments.
    """

    use_checkpoint_num_samples: bool = False
    """
    Use checkpoint to set the values of the total number of samples. If the sequence length and seed are the same, can be used to prevent reindexing of the data loader.
    """


@dataclass
class GlobalConfigLogging(GlobalConfigTemplate):
    """
    Logging Arguments
    """

    use_wandb: bool = None
    """Flag indicating if wandb is to be used."""

    wandb_group: str = None
    """Weights and Biases group name - used to group together "runs"."""

    wandb_run_name: str = None
    """Weights and Biases run name."""

    wandb_team: str = None
    """Team name for Weights and Biases."""

    wandb_project: str = "neox"
    """wandb project name"""

    wandb_host: str = "https://api.wandb.ai"
    """url of the wandb host"""

    wandb_init_all_ranks: bool = False
    """Initialize wandb on all ranks."""

    git_hash: str = get_git_commit_hash()
    """current git hash of repository"""

    log_dir: str = None
    """
    Directory to save logs to.
    """

    tensorboard_writer = None
    """
    initialized tensorboard writer
    """

    tensorboard_dir: str = None
    """
    Write TensorBoard logs to this directory.
    """

    log_interval: int = None
    """
    Interval between logging.
    """

    log_grad_pct_zeros: bool = False
    """
    Log the percentage of zeros for the gradient of each parameter to wandb / tensorboard (useful for debugging). Needs wandb_init_all_ranks set to True if using pipeline parallelism to log all ranks.
    """

    log_param_norm: bool = False
    """
    Log the frob norm of the parameters to wandb / tensorboard (useful for debugging). Needs wandb_init_all_ranks set to True if using pipeline parallelism to log all ranks.
    """

    log_grad_norm: bool = False
    """
    Log the frob norm of the gradients to wandb / tensorboard (useful for debugging).
    (N.B - this will only work with pp = 0 for now, as we don't have access to the gradients of the model because
    deepspeed.)
    """

    log_optimizer_states: bool = False
    """
    Log the frob norm of the optimizer states to wandb / tensorboard (useful for debugging).
    """

    log_gradient_noise_scale: bool = False
    """
    Whether to log the gradient noise scale when training (cf. https://arxiv.org/abs/1812.06162 for explanation)
    """

    gradient_noise_scale_n_batches: int = 5
    """
    Number of batches to accumulate gradients for in the gradient noise scale logger.
    """

    gradient_noise_scale_cpu_offload: bool = False
    """
    Whether to offload the buffered gradients to cpu when measuring gradient noise scale.
    """

    log_memory_stats: bool = False
    """
    Whether to track torch CUDA Caching Allocator memory stats.
    Helpful for diagnosing memory leaks, fragmentation and other memory issues
    that could lead to excessive calls to cudaMalloc / cudaFree which can slow down training.
    """

    log_memory_alloc_counts: bool = False
    """
    Log cuda memory allocation counts from `torch.cuda.memory_stats`.
    These stats are aggregated across all ranks and logged as summary stats (min, median, max, mean, stddev)

    """

    print_mem_alloc_stats: bool = False
    """
    Whether to print per-rank device mem alloc counts from `torch.cuda.memory_stats`.
    """

    mem_alloc_stats_ranks: int = 1
    """
    Print mem alloc stats only of rank % mem_alloc_stats_ranks == 0
    """

    deepspeed_enable_comms_logging: bool = False
    """
    Turn on deepspeed comms logger

    Uses deepspeed comms logger to gather sizes, times, and other stats for collective ops.

    NOTE: @jeromeku
    Use this flag instead of passing this argument to deepspeed_config to enable more fine-grained control over the logging.
    See [here](https://github.com/microsoft/DeepSpeed/blob/5c4b97f1092b798508dab4321b2ac79a9f554e72/docs/_tutorials/comms-logging.md) for details.
    """

    deepspeed_comms_print_summary: bool = False
    """
    Whether to print comms log summary at the end log rank interval
    """

    deepspeed_comms_ranks_to_log: int = 1
    """
    Which logs to print serialized comms dict, every `ranks_to_log` rank will print if `rank % ranks_to_log == 0`
    """

    deepspeed_comms_ops_to_log: list = None
    """
    Which collective ops to log (e.g., 'all_reduce', 'all_gather_into_tensor', etc.)
    None means all ops
    """
    deepspeed_comms_logging_interval: int = 1
    """
    How often to log comms stats. Default is 1 (every step).
    NOTE: @jeromeku
    Logging comms is very expensive as it synchronizes after every collective op, only use for debugging or to get a sense of how the system is performing.
    """

    deepspeed_comms_logging_verbose: bool = False
    """
    Enable verbose comms logging, in which case details of each comms op are logged after they are done.
    """

    deepspeed_comms_logging_debug: bool = False
    """
    Print additional debug information when logging comms such as the calling function.
    """

    debug_dir: str = None
    """
    Directory to save debug information to.
    """

    debug_print: bool = False
    """
    Print debugging info such as layer types and activation shapes.
    """

@dataclass
class GlobalConfigProfiler(GlobalConfigTemplate):
    """
    `torch.profiler` Arguments
    """

    should_profile: bool = False
    """
    Whether to enable profiling.
    """

    profiler_type: Literal["torch", "nsys", "none"] = "none"
    """
    Profiler type, will impact how ranges are profiled (`nvtx` vs `torch.profiler.record_function`).

    Note that `nsys` needs to be launched externally.
    """
    profile_ranks: list = None
    """
    Which ranks should profile
    """
    # ---- Torch profiler options ---- #
    profile_cpu: bool = True
    """
    Whether to profile CPU activities.
    """

    profile_cuda: bool = True
    """
    Whether to profile CUDA activities.
    """

    profile_memory: bool = False
    """
    Whether to enable memory profiling.
    """

    profile_with_stack: bool = False
    """
    Whether to enable stack profiling.
    """

    profile_record_shapes: bool = False
    """
    Whether to enable shape profiling.
    """

    profile_with_flops: bool = False
    """
    Whether to enable flops profiling.
    """

    profiler_schedule_wait: int = 4
    """
    The number of iterations to wait before profiling.
    """

    profiler_schedule_warmup: int = 5
    """
    The number of iterations to warmup before profiling.
    """

    profiler_schedule_active: int = 1
    """
    The number of iterations of active (actual) profiling.
    """

    profiler_schedule_repeat: int = 0
    """
    The number of profiling cycles (maps to `repeat` kwarg of `torch.profiler.schedule`).
    """

    profiler_output_dir: str = "torchprofiler_traces"
    """
    The directory to save the profiling traces.
    """

    profiler_clean_output_dir: bool = False
    """
    Whether to clean the output directory before profiling.
    """

    profiler_num_rows: int = -1
    """
    Number of rows to dump in the profile key average table.
    """

    # ---- CUDA profiler options ---- #
    emit_nvtx: bool = True
    """
    Automatically add `nvtx` ranges to `torch.autograd` ops
    """

    disable_autograd_multithreading: bool = True
    """
    Run autograd in a single threaded manner.

    Helpful when profiling since NVTX ranges; if autograd
    runs on multithreaded contexts, nvtx ranges will not be
    captured correctly for backward passes.
    """

    nsys_stop_on_exit: bool = True
    """
    Explicitly stop the cuda profiler on exit.

    Note this can sometimes result in hangs when using in distributed context
    with DeepSpeed.
    """

    nsys_warmup_steps: int = 3
    """
    Number of iterations before starting profiling.
    """

    nsys_num_steps: int = 2
    """
    Number of profiling steps.
    """

    #nvidia-resiliency-ext
    use_next: bool = False
    """
    Use nvidia_resiliency_ext to detect stragglers, kernel and section summaries
    """

    next_gather_on_rank0: bool = True
    """
    Gather results from straggler.Detector on rank 0
    """

    next_report_interval: int = 10
    """
    Interval at which to print report
    """

    next_section_scores: bool = True
    next_section_rel_threshold: float = .7
    next_section_individual_threshold: float = .7

    next_gpu_scores: bool = True
    next_gpu_rel_threshold: float = .7
    next_gpu_individual_threshold: float = .7

    next_stragglers: bool = True

    # Heimdall straggler detector
    heimdall_log_straggler: bool = False
    """
    If set, tracks and logs straggler per GPU.
    """

    heimdall_log_interval: int = 10
    """
    Interval at which to log heimdall straggler results
    """

    heimdall_disable_straggler_on_startup: bool = True
    """
    If set, StragglerDetector is disabled on startup.
    """

    heimdall_straggler_port: int = 65535
    """
    Port number to toggle StragglerDetector on/off at runtime
    """

    heimdall_straggler_minmax_count: int = 32
    """
    Number of ranks to report with high/low estimated throughput
    """

    # ---- Memory snapshot options ---- #
    enable_memory_snapshot: bool = False
    """
    Whether to enable memory snapshot.
    """

    save_memory_snapshot_folder: str = "memory_snapshots"
    """
    Directory to save memory snapshots to.
    """

    memory_snapshot_rank0_only: bool = False
    """
    Whether to save memory snapshots only on rank 0.
    """

    memory_snapshot_freq: int = 3
    """
    Frequency of memory snapshots.
    """

    memory_snapshot_max_entries: int = 100000
    """
    Maximum number of entries to save in memory snapshots.
    """


@dataclass
class GlobalConfigOther(GlobalConfigTemplate):
    """
    Misc. Arguments
    """

    distributed_backend: str = "nccl"
    """
    Which backend to use for distributed training.
    """

    local_rank: int = None
    """
    local rank passed from distributed launcher.
    """

    rank: int = None
    """
    global rank of process being run (passed in via distributed launcher)
    """

    lazy_mpu_init: bool = False
    """
    If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead. Also turns on use-cpu-initialization flag. This is for external DDP manager.
    """

    short_seq_prob: float = 0.1
    """
    Probability of producing a short sequence.
    """

    eod_mask_loss: bool = False
    """
    Mask loss for the end of document tokens.
    """

    pad_mask_loss: bool = False
    """
    Mask loss for the padding tokens.
    """

    adlr_autoresume: bool = False
    """
    Enable auto-resume on adlr cluster.
    """

    adlr_autoresume_object = None
    """
    imported autoresume
    """

    adlr_autoresume_interval: int = 1000
    """
    Intervals over which check for auto-resume termination signal
    """

    seed: int = 1234
    """
    Random seed used for python, numpy, pytorch, and cuda.
    """

    onnx_safe: bool = False
    """
    Use workarounds for known problems with Torch ONNX exporter
    """

    deepscale: bool = False
    """
    (Deprecated) enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)'
    """

    deepscale_config: str = None
    """(Deprecated) deepscale json configuration file."""

    deepspeed_mpi: bool = False
    """
    Run via MPI, this will attempt to discover the necessary variables to initialize torch distributed from the MPI environment
    """

    deepspeed_slurm: bool = False
    """
    Run via SLURM, this will attempt to discover the necessary variables to initialize torch distributed from the SLURM environment
    """

    deepspeed_jsrun: bool = False
    """
    Run via JSRUN, this will attempt to discover the necessary variables to initialize torch distributed from the IBM LSF environment
    """

    use_srun_launcher: bool = False
    """
    Whether to use srun launcher rather tha `pdsh` to launch distributed training
    See `savanna/launcher/README`
    """

    srun_launcher_type: Literal["torch", "deepspeed", "srun"] = None
    """
    Type of distributed launcher - should not be set directly but rather through `launcher/generate_distributed_launcher.py`

    All are functionally equivalent
    - torch and deepspeed launch a single process on each node, which is then responsible for
    spawning number of local processes equal to the number of local GPUs
    - additionally, deepspeed enables per rank logging by redirecting stdout / stderr during
    local process creation
    - srun will spin up number of processes equal to world size which enables more control over
    process creation, useful for features such as NUMA binding
    """

    enable_each_rank_log: bool = False
    """
    Redirect the stdout and stderr from each rank into different log files

    enable_each_rank_log should be a directory that can be written to
    log files will be named "%Y%m%d%H%M%S_rank{rank}.log"
    """

    user_script: str = None
    """
    user script to be run
    """

    iteration: int = None
    """
    Set during training
    """

    do_train: int = None
    """
    Set during training
    """

    do_valid: int = None
    """
    Set during training
    """
        
    do_test: int = None
    """
    Set during training
    """

    save_iters: list = None
    """
    Set during training
    """

    global_num_gpus: int = None
    """
    Set during launching
    """


@dataclass
class GlobalConfigTokenizer(GlobalConfigTemplate):
    """
    Tokenizer Arguments
    """

    tokenizer_type: Literal[
        "GPT2BPETokenizer",
        "HFTokenizer",
        "HFGPT2Tokenizer",
        "SPMTokenizer",
        "CharLevelTokenizer",
        "TiktokenTokenizer",
    ] = "GPT2BPETokenizer"
    """
    Type of tokenizer to use - should be one of ["GPT2BPETokenizer", "HFTokenizer", "HFGPT2Tokenizer", "SPMTokenizer", "CharLevelTokenizer", "TiktokenTokenizer"]
    """

    padded_vocab_size: int = None
    """
    Total (padded) vocabulary size of tokenizer. Configured after launching of training,
    as it's dependent on the parallelism size.
    """

    tokenizer = None
    """
    tokenizer object loaded into memory and accessible by other functions
    """


@dataclass
class CheckpointStore(GlobalConfigTemplate):
    storage_type: str = "s3"
    access_id: str = None
    secret: str = None
    location: str = None
    num_checkpoints: int = None


@dataclass
class GlobalConfigTraining(GlobalConfigTemplate):
    """
    Training Arguments
    """

    data_path: str = None
    """
    Path to combined dataset to split.
    """

    use_shared_fs: bool = True
    """
    Whether to use a shared filesystem for data loading. If False, local rank 0 on all nodes will preprocess the data,
    otherwise only global rank 0 will preprocess the data. This is implemented in megatron/data/sequence_dataset.py::_build_index_mappings.
    """

    train_data_paths: list = None
    """
    List of paths to train datasets.
    """

    test_data_paths: list = None
    """
    List of paths to test datasets.
    """

    valid_data_paths: list = None
    """
    List of paths to validation datasets.
    """

    per_ds_valid_data_paths: list = None
    """
    List of paths to per dataset validation datasets, should be subset of valid_data_paths
    """
    
    train_data_weights: list = None
    """
    List of 'weights' that decide how often to sample from each training dataset when blending datasets. If None, defaults to equal weighting.
    Should be a list the same length as `train_data_paths`
    """

    valid_data_weights: list = None
    """
    List of 'weights' that decide how often to sample from each validation dataset when blending datasets. If None, defaults to equal weighting.
    Should be a list the same length as `valid_data_paths`
    """

    test_data_weights: list = None
    """
    List of 'weights' that decide how often to sample from each test dataset when blending datasets. If None, defaults to equal weighting.
    Should be a list the same length as `test_data_paths`
    """

    weight_by_num_documents: bool = False
    """
    If True, Builds dataset weights from a multinomial distribution over groups of data according to the number of
    documents in each group.

    WARNING: setting this to True will override any user provided weights

    We sample from a group according to the probability p(L) ∝ |L| ** α,
    where p(L) is the probability of sampling from a given group,
          |L| is the number of examples in that datapoint,
          and α is a coefficient that acts to upsample data from underrepresented groups

    Hence α (`alpha`) allows us to control how much to 'boost' the probability of training on low-resource groups.

    See https://arxiv.org/abs/1911.02116 for more details
    """

    weighted_sampler_alpha: float = 0.3
    """
    Alpha value for `weight_by_num_documents`. Only has an effect if `weight_by_num_documents` = True.

    when alpha = 1, the probability of sampling from a given group = n_samples / total_samples
    as alpha -> 0, the probability of sampling from all groups becomes equal, and number of documents has no effect
    as alpha -> inf, the probability of sampling from the groups with *the most samples* -> 1
    """

    data_impl: str = "infer"
    """
    Implementation of indexed datasets.
    """

    enforce_sample_length: bool = False
    """
    Enforces that all samples are equal to sequence length.
    """

    mmap_warmup: bool = False
    """
    Warm up mmap files.
    """

    save: str = None
    """
    Output directory to save checkpoints to.
    """

    checkpoint_stores: list = None
    """
    Location to upload the checkpoint before local cleanup
    """

    async_save: bool = False
    """
    Knob to enable async checkpoint writing
    """

    config_files: dict = None
    """
    Store of original config files mapping config filename to file contents
    """

    load: str = None
    """
    Directory containing a model checkpoint.
    """

    checkpoint_validation_with_forward_pass: bool = False
    """
    save input and output of a forward pass with the checkpoint and validate after load
    """

    checkpoint_scale: Literal["linear", "log"] = "linear"
    """
    How step at which checkpoints are saved should scale. "linear" implies 1 checkpoint will be saved at every multiple of `checkpoint-factor`,
    while "log" implies that the number of steps between each checkpoint will be multiplied by `checkpoint-factor` at each step, starting from step 1.
    """

    checkpoint_factor: int = None
    """
    Acts as a multiplier on either the "log" or "linear" checkpoint spacing.

    With `checkpoint-scale="linear"`, `checkpoint-factor=20`, and `train-iters=100`, checkpoints will be saved at
    steps [20, 40, 60, 80, 100].

    With `checkpoint-scale="log"`, `checkpoint-factor=2`, and `train-iters=100`, checkpoints will be saved at
    steps [1, 2, 4, 8, 16, 32, 64, 100].

    Note that the last checkpoint step is always saved.
    """

    extra_save_iters: list = None
    """
    Additional iterations when a checkpoint should be saved.
    Must be a list of ints or `None`.
    """

    checkpoint_strict_load: bool = True
    """
    Enable or disable strict loading of checkpoints
    NOTE: Be careful when setting this to False as it may result in uncaught errors
    """

    no_save_optim: bool = False
    """
    Do not save current optimizer.
    """

    no_save_rng: bool = False
    """
    Do not save current rng state.
    """

    no_load_optim: bool = False
    """
    Do not load optimizer when loading checkpoint.
    """

    no_load_rng: bool = False
    """
    Do not load rng state when loading checkpoint.
    """

    finetune: bool = False
    """
    Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0. Assumed when loading a release checkpoint.
    """

    warmstart: bool = False
    """
    Whether to warmstart the model from a previous checkpoint at iteration 0.
    """

    alignment_method: str = None
    """
    Method for aligning the model with a provided reward.
    """

    dpo_beta: float = 1.0
    """
    Beta parameter for Direct Preference Optimization (DPO).
    """

    dpo_data_seq_length: float = None
    """
    Sequence length for DPO dataloading.
    """

    batch_size: int = None
    """
    training microbatch size per gpu
    """

    train_iters: int = None
    """
    Number of iterations to run for training.
    """

    eval_iters: int = 100
    """
    Number of iterations to run for evaluation validation/test for.
    """
    
    do_per_ds_valid: bool = False
    """
    Whether to run evaluation on each subset dataset.
    """

    eval_per_ds_interval: int = None
    """
    Interval between running evaluation on each subset dataset.
    """

    eval_per_ds_iters: int = None
    """
    Number of iterations to run for per dataset evaluation validation run.
    """

    num_per_ds_evals: int = 0
    """
    Number of per dataset evaluation datasets.  Set automatically in `savanna/data/data_utils.py` based on the number of validation dataset paths provided.
    """
    
    train_data_token_index: int = None
    """
    Index of the data to initialize the training sampler in units of tokens. If not specified, the index will be automatically computed from iteration, global batch size, gradient accumulation steps, and sequence length.
    """

    train_val_test_num_samples: List[int] = None
    """
    Stores the number of train, validation, and test samples.
    """

    keep_last_n_checkpoints: int = None
    """
    Number of last checkpoints to keep
    """

    eval_interval: int = 1000
    """
    Interval between running evaluation on validation set.
    """

    split: str = "969, 30, 1"
    """
    Comma_separated list of proportions for training, validation, and test split. For example the split 90,5,5 will use 90% of data for training, 5% for validation and 5% for test.
    """

    vocab_file: str = None
    """
    Path to the vocab file.
    """

    merge_file: str = None
    """
    Path to the BPE merge file.
    """

    num_workers: int = 2
    """
    Dataloader number of workers.
    """

    exit_interval: int = None
    """
    Exit the program after the iteration is divisible by this value.
    """

    attention_dropout: float = 0.1
    """
    Post attention dropout probability.
    """

    hidden_dropout: float = 0.1
    """
    Dropout probability for hidden state transformer.
    """

    weight_decay: float = 0.01
    """
    Weight decay coefficient for L2 regularization.
    """

    checkpoint_activations: bool = False
    """
    Checkpoint activation to allow for training with larger models, sequences, and batch sizes.
    """

    checkpoint_num_layers: int = 1
    """
    Chunk size (number of layers) for checkpointing.
    """

    deepspeed_activation_checkpointing: bool = True
    """
    DEPRECATED - TODO: remove
    Uses activation checkpointing from deepspeed
    """

    contiguous_checkpointing: bool = False
    """
    Contiguous memory checkpointing for activations.
    """

    checkpoint_in_cpu: bool = False
    """
    Move the activation checkpoints to CPU.
    """

    synchronize_each_layer: bool = False
    """
    does a synchronize at the beginning and end of each checkpointed layer.
    """

    profile_backward: bool = False
    """
    Enables backward pass profiling for checkpointed layers.
    """

    partition_activations: bool = False
    """
    Partition Activations across GPUs before checkpointing.
    """

    gas: int = None
    """gradient_accumulation_steps"""  # TODO this is a duplicate, remove?

    clip_grad: float = None
    """
    Gradient clipping based on global L2 norm.
    """

    hysteresis: int = 2
    """
    hysteresis for dynamic loss scaling
    """

    dynamic_loss_scale: bool = None
    """
    flag indicating whether dynamic loss scale is used
    """

    loss_scale: float = None
    """
    Static loss scaling, positive power of 2
    values can improve fp16 convergence. If None, dynamic loss scaling is used.
    """

    loss_scale_window: float = 1000.0
    """
    Window over which to raise/lower dynamic scale.
    """

    min_scale: float = 1.0
    """
    Minimum loss scale for dynamic loss scale.
    """

    char_level_ppl: bool = False
    """
    Whether to calculate character level perplexity as well as token level perplexity. (may incur a time cost)
    """

    use_mup: bool = False
    """
    Whether to use Microsoft's Mup https://github.com/microsoft/mup
    """

    coord_check: bool = False
    """
    Whether to generate a "coord check" plot to verify mup's implementation in neox
    """

    save_base_shapes: bool = False
    """
    Whether to save base shapes for mup. This will save the shapes to the path specified in base-shapes-file.
    """

    base_shapes_file: str = None
    """
    Path to the base shapes to save to/load from
    """

    mup_init_scale: float = 1.0
    """
    Initialization scale: All the parameters are multiplied by this value
    """

    mup_attn_temp: float = 1.0
    """
    Attention temperature: Reciprocal of the multiplier applied to the input to attention softmax
    """

    mup_output_temp: float = 1.0
    """
    Output temperature: Reciprocal of the multiplier applied to the input to softmax that
    produces the distribution over output tokens.
    """

    mup_embedding_mult: float = 1.0
    """
    Scalar by which we multiply the output of the embedding layer
    """

    mup_rp_embedding_mult: float = 1.0
    """
    Scalar by which we multiply vectors representing relative position
    """

    mup_width_scale: int = 2
    """
    What to scale width by when creating the delta model for mup
    """


@dataclass
class GlobalConfigTextgen(GlobalConfigTemplate):
    """
    Text Generation arguments
    """

    text_gen_type: str = None
    """
    How to generate text/sample the model.
    Options: `unconditional`, `input-file`, `interactive`
    """

    temperature: float = 0.0
    """
    exponential scaling output distribution ("higher == more risk")
    """

    top_p: float = 0.0
    """
    Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    """

    top_k: int = 0
    """
    integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    """

    return_logits: bool = False
    """
    Boolean for whether to return the logits for generated tokens
    """

    maximum_tokens: int = 64
    """
    maximum number of tokens to be generated
    """

    prompt_end: str = "\n"
    """
    a single prompt's end. Defaults to newline
    """

    sample_input_file: str = None
    """
    Get input from file instead of interactive mode, each line is an input.
    """

    sample_output_file: str = "samples.txt"
    """
    Output file
    """

    num_samples: int = 1
    """
    Number of samples to generate unconditionally, defaults to 1 and interactive conditional sampling
    """

    recompute: bool = False
    """
    During generation recompute all attention instead of using previously computed keys/values.
    Should be set to true for sparse attention models
    """

    eval_results_prefix: str = ""
    """
    prefix to which to save evaluation results - final fp will be {eval_results_prefix}_eval_results_yy-mm-dd-HH-MM.json
    """

    eval_tasks: list = None
    """
    Tasks to evaluate on using lm_eval_harness
    """

    save_retain_interval: int = 2000
    """
    Checkpoint intervals to be uploaded. In the iteration of a saved checkpoint is divisible by `save_retain_interval` then it will be uploaded.
    """    """
    Checkpoint intervals to be uploaded. In the iteration of a saved checkpoint is divisible by `save_retain_interval` then it will be uploaded.
    """