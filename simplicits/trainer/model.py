import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=24, num_layers=9):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ELU())
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ELU())
        self.layers.append(nn.Linear(hidden_size, output_size))
        self.output_size = output_size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
