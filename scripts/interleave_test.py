import torch
from einops import rearrange


def legacy_interleave(x, num_groups_per_tp_rank):
    x1 = rearrange(x[...,0::3], "b l (g dg) -> b l g dg", g=num_groups_per_tp_rank)
    x2 = rearrange(x[...,1::3], "b l (g dg) -> b l g dg", g=num_groups_per_tp_rank)
    v  = rearrange(x[...,2::3], "b l (g dg) -> b l g dg", g=num_groups_per_tp_rank)
    return x1, x2, v

def new_interleave(x, num_groups_per_tp_rank):
    x1, x2, v = rearrange(
        x,
        'b l (g dg p) -> b l g p dg',
        p=3,
        g=num_groups_per_tp_rank
    ).unbind(dim=3)
    return x1, x2, v

if __name__ == "__main__":
    for d, num_groups, mp_size in ((4096, 256, 2), (4096, 256, 4), (4096, 256, 8), (8192, 512, 2), (8192, 512, 4), (8192, 512, 8)):
        bs = 8
        seqlen = 8192

        D = 3 * d // mp_size
        x = torch.randn(bs, seqlen, D)
        num_groups_per_tp_rank = num_groups // mp_size

        print(f"Checking shape: {mp_size=} {bs=}, {seqlen=}, {d=} {D=} {num_groups=} {num_groups_per_tp_rank=}")

        _x1, _x2, _v = legacy_interleave(x, num_groups_per_tp_rank)
        x1, x2, v = new_interleave(x, num_groups_per_tp_rank)
        assert torch.equal(_x1, x1), "x1 mismatch"
        assert torch.equal(_x2, x2), "x2 mismatch"
        assert torch.equal(_v, v), "v mismatch"
    print("All tests passed!")
