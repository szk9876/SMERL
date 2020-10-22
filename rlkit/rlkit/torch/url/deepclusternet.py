import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule


def identity(x):
    return x


class DeepClusterNet(PyTorchModule):    
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=nn.ReLU,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
    ):
        self.save_init_params(locals())
        super().__init__()
        
        self.init_w = init_w

        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.encoder_layers = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.encoder_layers.append(fc)
            self.encoder_layers.append(self.hidden_activation())
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.top_layer = None
        self.to(ptu.device)

    def top_layer_init(self):
        self.top_layer = nn.Linear(in_features=self.hidden_sizes[-1], out_features=self.output_size)
        self.top_layer.weight.data.uniform_(-self.init_w, self.init_w)
        self.top_layer.bias.data.uniform_(-self.init_w, self.init_w)

    def forward(self, input):
        h = self.encoder(input)
        if self.top_layer:
            h = self.top_layer(h)
        output = h
        return output
        
    def prepare_for_inference(self):
        self.top_layer = None
        assert type(list(self.encoder.children())[-1]) is nn.ReLU
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])   # remove last ReLU in encoder
        self.to(ptu.device)
        self.eval()
        
    def prepare_for_training(self):
        assert self.top_layer is None
        encoder = list(self.encoder.children())
        encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)
        self.top_layer_init()
        self.to(ptu.device)
        self.train()
        

if __name__ == '__main__':
    ptu.set_gpu_mode(True)

    dcnet = DeepClusterNet(hidden_sizes=[16, 16, 16],
                           output_size=20,
                           input_size=8,
                           )


    import ipdb
    ipdb.set_trace()
    x = 1