import math
import torch
import torch.nn as nn

from einops import rearrange, repeat


has_pykeops = False
from savanna.ops.vandermonde import log_vandermonde_naive as log_vandermonde


_c2r = torch.view_as_real
_r2c = torch.view_as_complex

if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()


class OptimModule(nn.Module):
    """Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters"""

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class LaughingHyenaModalFilter(nn.Module):
    def __init__(self, global_config, d_model, order, theta_max):
        super().__init__()
        self.order = order
        self.d_model = d_model

        # Init poles and residues
        self.register_parameter("r", nn.Parameter(torch.ones(order // 2, d_model)))
        self.register_parameter("theta", nn.Parameter(torch.ones(order // 2, d_model)))
        self.register_parameter("R_re", nn.Parameter(torch.ones(order // 2, d_model)))
        self.register_parameter("R_im", nn.Parameter(torch.ones(order // 2, d_model)))

        self.register_parameter("h_0", nn.Parameter(torch.ones(1, d_model)))
        r_min, r_max = (
            global_config.hyena_filter_r_min,
            global_config.hyena_filter_r_max,
        )
        self._init_params(r_max, r_min, theta_max)

    def _init_params(self, r_max, r_min, theta_max):
        # Init poles distributed uniformly of ring of the complex plane
        # between r_min and r_max and phase between 0 and theta_max
        u1 = torch.rand(self.order // 2, self.d_model)
        u2 = torch.rand(self.order // 2, self.d_model)
        self.r.data = r_min + (r_max - r_min) * u1
        self.theta.data = theta_max * u2
        # Init residues with Glorot initialization
        self.R_re.data = torch.randn(self.order // 2, self.d_model) * math.sqrt(2 / self.order)
        self.R_im.data = torch.randn(self.order // 2, self.d_model) * math.sqrt(2 / self.order)

    def _get_poles_and_residues(self):
        # poles
        p = self.r * torch.exp(1j * self.theta)
        # residues
        R = self.R_re + 1j * self.R_im
        return p, R

    def compute_filter(self, L):
        p, R = self._get_poles_and_residues()
        t = torch.arange(L - 1).unsqueeze(1).unsqueeze(2).to(p)
        h = torch.sum(R * p**t, dim=1).real
        return h

    def forward(self, L, *args, **kwargs):
        # evaluate filter for t = 1, ..., L
        h = self.compute_filter(L)
        # stack h_0 to the beginning of the filter
        h = torch.cat([self.h_0.to(h), h], dim=0)
        h = rearrange(h, "L D -> D L")
        return h


class ImplicitComplexModalFilter(OptimModule):

    def __init__(
        self,
        A,
        B,
        C,
        log_dt,
        L=None,
        disc="bilinear",
        real_type="exp",
        lr=None,
        bandlimit=None,
    ):
        super().__init__()
        self.L = L
        self.disc = disc
        self.bandlimit = bandlimit
        self.real_type = real_type

        assert A.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = A.size(-1)
        assert A.size(-2) == B.size(-2)
        assert self.H % A.size(-2) == 0
        self.n_ssm = A.size(-2)
        self.repeat = self.H // A.size(0)

        self.channels = C.shape[0]
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))

        if lr is None or isinstance(lr, float):
            lr_dict = {}
        else:
            lr_dict, lr = lr, None

        self.register("log_dt", log_dt, lr_dict.get("dt", lr))
        self.register("B", _c2r(B), lr_dict.get("B", lr))
        self.register("inv_A_real", torch.log(-A.real), lr_dict.get("A", lr))
        self.register("A_imag", A.imag.contiguous(), lr_dict.get("A", lr))

    def forward(self, L, state=None, u=None, **kwargs):
        log_dt = self.log_dt.to(torch.float32)
        dt = torch.exp(log_dt)
        C = _r2c(self.C.to(torch.float32))  # (C H N)

        A_real = -torch.exp(self.inv_A_real)
        A_imag = self.A_imag
        A = A_real + 1j * self.A_imag

        B = _r2c(self.B.to(torch.float32))
        B = repeat(B, "t n -> 1 (v t) n", v=self.repeat)

        A = repeat(A, "t n -> (v t) n", v=self.repeat)
        dtA = A * dt.unsqueeze(-1)  # (H N)

        C = (B[:, None, :, :] * C).view(-1, self.H, self.N)

        C = C * (1.0 - dtA / 2).reciprocal() * dt.unsqueeze(-1)  # or * dtA / A
        dA = (1.0 + dtA / 2) / (1.0 - dtA / 2)
        dA_log = repeat(dA.log(), "h d -> (c h) d", c=C.shape[0])
        K = log_vandermonde(C, dtA, L)  # (H L)

        K = K.view(-1, self.channels, self.H, L)  # (1+B C H L)
        K = K[-1, :, :, :]  # (C H L)
        return K, None
