import torch
from torch import Tensor
from torch.nn import Module, Linear


class NetVAE(Module):
    def __init__(self, module_features: tuple):
        super(NetVAE, self).__init__()

        self.encoder, self.decoder = [], []

        for i in range(len(module_features) - 2):
            in_features, out_features = module_features[i], module_features[i + 1]
            fc_encoder = Linear(in_features, out_features)
            fc_decoder = Linear(out_features, in_features)

            self.encoder.append(fc_encoder)
            self.decoder.insert(0, fc_decoder)

        in_features, out_features = module_features[-2:]

        self.fc_mean = Linear(in_features, out_features)
        self.fc_log_var = Linear(in_features, out_features)
        self.decoder.insert(0, Linear(out_features, in_features))
        self.fc_last = self.decoder[-1]

    def encode(self, x):
        for fc in self.encoder:
            x = torch.relu(fc(x))

        return self.fc_mean(x), self.fc_log_var(x)

    def decode(self, z):
        for fc in self.decoder[0: -1]:
            z = torch.relu(fc(z))

        return torch.sigmoid(self.fc_last(z))

    def reparametrization(self, mean, log_var):
        std = 0.5 * torch.exp(log_var)
        z = torch.randn(std.size()) * std + mean

        return z

    def forward(self, x) -> (Tensor, Tensor, Tensor):
        mean, log_var = self.encode(x)
        z = self.reparametrization(mean, log_var)

        return self.decode(z), mean, log_var
