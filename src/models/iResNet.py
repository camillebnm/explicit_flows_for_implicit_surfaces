import torch
import numpy as np
import torch.nn as nn


def fixed_point_invert(g, y, iters=15, verbose=False):

        x = y
        dim = x.size(-1)
        for i in range(iters):
            x = y - g(x)
            if verbose:
                err = (y - (x + g(x))).view(-1, dim).norm(dim=-1).mean()
                err = err.detach().cpu().item()
                print("iter:%d err:%s" % (i, err))
        return x


class iResNet(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.dim = 4
        self.out_dim = 3
        self.hidden_size = 256
        self.n_blocks = 6
        self.n_g_blocks = 1

        # Network modules
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(
                InvertibleResBlockLinear(
                    self.dim, self.hidden_size,
                    nblocks=self.n_g_blocks, nonlin="elu",
                    pos_enc_freq=5,
                )
            )

    def forward_b(self, x, t):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        out = x
        for block in self.blocks:
            out = block(out, t)
        return out, None

    def invert(self, y, t, verbose=False, iters=15):
        x = y
        for block in self.blocks[::-1]:
            x = block.invert(x, t, verbose=verbose, iters=iters)
        return x, None

    def forward(self, x, t): 
        xx = x[:,:-1]
        t0 = x[:,-1:]
        xx, _ = self.invert(xx,t0)

        xx, _ = self.forward_b(xx, t+t0)
        return {"model_out" : torch.cat((xx,t+t0),axis=-1), "model_in" : x}

class LipBoundedPosEnc(nn.Module):

    def __init__(self, inp_features, n_freq, cat_inp=True):
        super().__init__()
        self.inp_feat = inp_features
        self.n_freq = n_freq
        self.cat_inp = cat_inp
        self.out_dim = 2 * self.n_freq * self.inp_feat
        if self.cat_inp:
            self.out_dim += self.inp_feat

    def forward(self, x):
        """
        :param x: (bs, npoints, inp_features)
        :return: (bs, npoints, 2 * out_features + inp_features)
        """
        assert len(x.size()) == 3
        bs, npts = x.size(0), x.size(1)
        const = (2 ** torch.arange(self.n_freq) * np.pi).view(1, 1, 1, -1)
        const = const.to(x)

        # Out shape : (bs, npoints, out_feat)
        cos_feat = torch.cos(const * x.unsqueeze(-1)).view(
            bs, npts, self.inp_feat, -1)
        sin_feat = torch.sin(const * x.unsqueeze(-1)).view(
            bs, npts, self.inp_feat, -1)
        out = torch.cat(
            [sin_feat, cos_feat], dim=-1).view(
            bs, npts, 2 * self.inp_feat * self.n_freq)
        const_norm = torch.cat(
            [const, const], dim=-1).view(
            1, 1, 1, self.n_freq * 2).expand(
            -1, -1, self.inp_feat, -1).reshape(
            1, 1, 2 * self.inp_feat * self.n_freq)

        if self.cat_inp:
            out = torch.cat([out, x], dim=-1)
            const_norm = torch.cat(
                [const_norm, torch.ones(1, 1, self.inp_feat).to(x)], dim=-1)

            return out / const_norm / np.sqrt(self.n_freq * 2 + 1)
        else:

            return out / const_norm / np.sqrt(self.n_freq * 2)


class InvertibleResBlockLinear(nn.Module):

    def __init__(self, inp_dim, hid_dim, nblocks=1,
                 nonlin='leaky_relu',
                 pos_enc_freq=None):
        super().__init__()
        self.dim = inp_dim
        self.nblocks = nblocks

        self.pos_enc_freq = pos_enc_freq
        if self.pos_enc_freq is not None:
            inp_dim_af_pe = self.dim * (self.pos_enc_freq * 2 + 1)
            self.pos_enc = LipBoundedPosEnc(self.dim, self.pos_enc_freq)
        else:
            self.pos_enc = lambda x: x
            inp_dim_af_pe = inp_dim

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.utils.spectral_norm(
            nn.Linear(inp_dim_af_pe, hid_dim)))
        for _ in range(self.nblocks):
            self.blocks.append(
                nn.utils.spectral_norm(
                    nn.Linear(hid_dim, hid_dim),
                )
            )
        self.blocks.append(
            nn.utils.spectral_norm(
                nn.Linear(hid_dim, self.dim-1),
            )
        )

        self.nonlin = nonlin.lower()
        if self.nonlin == 'leaky_relu':
            self.act = nn.LeakyReLU()
        elif self.nonlin == 'relu':
            self.act = nn.ReLU()
        elif self.nonlin == 'elu':
            self.act = nn.ELU()
        elif self.nonlin == 'softplus':
            self.act = nn.Softplus()
        else:
            raise NotImplementedError

    def forward_g(self, x, t):
        x = torch.cat((x, t*torch.ones(x.shape[0], 1, device=x.device)), -1)
        orig_dim = len(x.size())
        if orig_dim == 2:
            x = x.unsqueeze(0)

        y = self.pos_enc(x)
        for block in self.blocks[:-1]:
            y = self.act(block(y))
        y = self.blocks[-1](y)

        if orig_dim == 2:
            y = y.squeeze(0)

        return y

    def forward(self, x, t):
        return x + self.forward_g(x, t)

    def invert(self, y, t, verbose=False, iters=15):
        return fixed_point_invert(
            lambda x: self.forward_g(x,t), y, iters=iters, verbose=verbose
        )
