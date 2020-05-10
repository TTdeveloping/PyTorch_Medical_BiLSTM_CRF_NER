import torch
import torch.nn as nn
import numpy as np


def init_embedding(input_embedding, seed=666):
    """初始化embedding层权重
    """
    torch.manual_seed(seed)
    scope = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -scope, scope)


def init_linear_weight_bias(input_linear, seed=1337):
    """
    :param input_linear:
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    nn.init.xavier_uniform_(input_linear.weight)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + 1))
    if input_linear.bias is not None:
        input_linear.bias.data.uniform_(-scope, scope)