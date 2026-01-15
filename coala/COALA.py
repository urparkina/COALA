import torch
import torch.nn as nn
import torch.nn.functional as F
import coala.Linear as Linear
import torch

from peft.tuners.tuners_utils import BaseTunerLayer


class COALA_Layer(nn.Module, BaseTunerLayer):
    def __init__(
            self,
            pre_layer: nn.Module,
            in_features: int,
            out_features: int,
            ratio: int,
            compress_strategy: str = 'empty',
            fp16: bool = False,
            params: dict = None,
            X = None,
        ):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        
        base_tensor = pre_layer.weight
        self.adapter = Linear.COALA_Linear(ratio, compress_strategy, fp16, params, base_tensor, X)


    def forward(self, x: torch.Tensor):
        return self.adapter(x)