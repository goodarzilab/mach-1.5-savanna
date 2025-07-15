import torch
from einops import rearrange


def legacy_split(x, num_groups_per_tp_rank):
    x = rearrange(x, "b l (g dg) -> b l g dg", g=3 * num_groups_per_tp_rank)
    x1, x2, v = torch.split(x, num_groups_per_tp_rank, dim=-2)
    return x1, x2, v

def new_split(x, num_groups_per_tp_rank):
    x = rearrange(x, "b l (g dg) -> b l g dg", g=3 * num_groups_per_tp_rank)
    x1, x2, v = rearrange(x, 'b l (n p) f -> b l n p f', p=3).unbind(dim=3)
    return x1, x2, v

def new_split_2(x, num_groups_per_tp_rank):
    #x = rearrange(x, "b l (g dg) -> b l g dg", g=3 * num_groups_per_tp_rank)
    x1, x2, v = rearrange(x, 'b l (g p dg) -> b l g p dg', g=num_groups_per_tp_rank, p=3).unbind(dim=3)
    return x1, x2, v

def new_split_3(x, num_groups_per_tp_rank):
    x1, x2, v = rearrange(x, 'b l (p g dg) -> b l p g dg', p=3, g=num_groups_per_tp_rank).unbind(dim=2)
    return x1, x2, v

if __name__ == "__main__":
    for d, num_groups, mp_size in ((4096, 256, 2), (4096, 256, 4), (4096, 256, 8), (8192, 512, 2), (8192, 512, 4), (8192, 512, 8)):
        bs = 8
        seqlen = 8192

        D = 3 * d // mp_size
        x = torch.randn(bs, seqlen, D)
        num_groups_per_tp_rank = num_groups // mp_size
#        x = rearrange(x, "b l (g dg) -> b l g dg", g=3 * num_groups_per_tp_rank)

        print(f"Checking shape: {mp_size=} {bs=}, {seqlen=}, {d=} {D=} {num_groups=} {num_groups_per_tp_rank=}")
        algo_1 = legacy_split
        algo_2 = new_split_3

        _x1, _x2, _v = algo_1(x, num_groups_per_tp_rank)
        x1, x2, v = algo_2(x, num_groups_per_tp_rank)
        assert torch.equal(_x1, x1), "x1 mismatch"
        assert torch.equal(_x2, x2), "x2 mismatch"
        assert torch.equal(_v, v), "v mismatch"
    print("All tests passed!")
