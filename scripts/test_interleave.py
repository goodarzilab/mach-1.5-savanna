# %%
import torch

bs = 1
seq_len = 8192
d = 4096
D = 3 * d
mp_size = 1
num_groups = 256

groups_per_tp_rank = num_groups // mp_size


# %%
def ref_interleave(x, cached=False):
    if cached:
        x1 = x[...,0::3]
        x2 = x[...,1::3]
        v  = x[...,2::3]
    else:
        x1 = x[:,0::3,:]
        x2 = x[:,1::3,:]
        v  = x[:,2::3,:]
    return x1, x2, v

def interleave(x, cached=False):
    if cached:
        x2 = x.reshape(bs, seq_len, D // 3, 3)
        x3 = x2.permute(0, 1, 3, 2)
        x4 = x3.unbind(dim=-2)
    else:
        x2 = x.reshape(bs, D // 3, 3, seq_len)
        x3 = x2.permute(0, 2, 1, 3)        
        x4 = x3.unbind(dim=1)
    return x4

def test_interleave(cached=False):
    
    x = torch.randn(bs, D, seq_len, device='cuda')
    if cached:
        x = x.permute(0, 2, 1)
    
    x1_ref, x2_ref, v_ref = ref_interleave(x, cached)
    
    
    x1, x2, v = interleave(x, cached)

    for ref, test in zip([x1_ref, x2_ref, v_ref], [x1, x2, v]):
        assert torch.allclose(ref, test)

    print(f"Interleave {cached=} all good!")

if __name__ == "__main__":
    test_interleave(cached=False)
    test_interleave(cached=True)

# %%
