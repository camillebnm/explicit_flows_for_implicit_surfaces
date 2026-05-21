# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD
from src.models.Siren import SIREN
from src.models.StaticRealNVP import StatRealNVP
# -----------------------------------------------------------------------------
# Invertible Layers



class Flow(nn.Module):
    """Class for the Flows Psi"""

    def __init__(self):
        super().__init__()

    def forward(self, t, x):
        raise NotImplementedError

    def vector_field(self, x):
        raise NotImplementedError

    def vector_field_jacobian(self, x):
        raise NotImplementedError





def general_linear(x):
    return x

# Dictionary of possible matrix constraints
LIE_DICT = {
    "general_linear": general_linear,
}


class AffineFlow(Flow):
    """Linear Flow with translation.
    Parameters:
        dim (int): The dimension of the affine transformation matrix.
        lie_group (str, optional): The type of Lie group to use for the affine transformation.
            Defaults to 'general_linear'.
    """

    def __init__(self, dim, *, A=None, b=None, lie_group="general_linear"):
        super().__init__()

        self.lie_group = LIE_DICT[lie_group]

        if A is None:
            A = torch.randn(dim, dim)
            A = nn.init.xavier_normal_(A)
            A = A / (10 * dim)
        else:
            assert (
                A.shape[0] == dim and A.shape[1] == dim
            ), "A must be a square matrix of size dim"
            A = A
        if b is None:
            b = torch.randn(dim, 1) / 10
        else:
            assert b.shape[0] == dim, "b must be a vector of length dim"
            b = b

        self.A = nn.Parameter(A)
        self.b = nn.Parameter(b)
        self.register_buffer("zeros", torch.zeros(1, dim + 1))

    def forward(self, t, x, mode="batched"):
        A = self.lie_group(self.A)
        b = self.b
        A = torch.cat((A, b), dim=1)
        A = torch.cat((A, self.zeros), dim=0)
        A = A.unsqueeze(0)

        t = t.view(-1, 1, 1)

        x = nn.functional.pad(x, (0, 1), "constant", 1.0)
        x = x.unsqueeze(2)

        At = A * t
        ex = torch.matrix_exp(At)

        if mode == "batched":
            x = torch.bmm(ex, x).squeeze()
        elif mode == "single":
            x = torch.matmul(ex, x).squeeze()
        return x[..., :-1]
        return x[..., :-1]

    def vector_field(self, x):
        A = self.lie_group(self.A)
        return (
            torch.matmul(A.unsqueeze(0), x.unsqueeze(2)) + self.b.unsqueeze(0)
        ).squeeze(2)

    def vector_field_jacobian(self, x):
        A = self.lie_group(self.A)
        return A.unsqueeze(0).expand(x.shape[0], -1, -1)



# -----------------------------------------------------------------------------


MLP_ACTIVATIONS = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}



class NeuralConjugate(nn.Module):
    """Constructs a conjugation from invertible networks.

    Parameters
    ---------
    layers: collection of layers
        List (or tuple) of layers.
    psi: str, optional
        Type of flow to use. Must be keys of `PSI_DICT`. Uses `matrix_exp` by
        default.
    """

    def __init__(self, H, psi):
        super(NeuralConjugate, self).__init__()
        # Choose the conjugate flow
        self.psi = psi
        self.H = H

    def forward(self, x, t, preserve_grad=True):
        #if not preserve_grad:
            #tx = tx.clone().detach().requires_grad_(True)
        y = x[:, :-1]
        t0 = x[:,-1:]
        t = t*torch.ones(x.shape[0], 1, device=x.device)

        y, _ = self.H.forward(y)
        y = self.psi(t, y)
        y, _ = self.H.inverse(y)

        return {"model_in": x, "model_out": torch.cat((y,t0+t), -1) }

    @staticmethod
    def load_from_config():


        H = StatRealNVP(style ="siren")


        psi = AffineFlow(dim=3, lie_group="general_linear")
        model = NeuralConjugate(H=H, psi=psi)
        return model

    def load_weights_from_model_path(self, model_path, device):
        """Loads weights from a file.

        # NOT IMPLEMENTED YET
        # TODO: Implement this method

        Parameters
        ----------
        model_path: str, PathLike

        device: str, torch.device

        Returns
        -------
        self: NeuralConjugate

        Raises
        ------
        FileNotFoundError if the file pointed by `model_path` is missing.
        """
        raise NotImplementedError()

