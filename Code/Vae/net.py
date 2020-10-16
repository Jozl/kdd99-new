import torch
from torch import nn, Tensor


class NetVAE(nn.Module):
    def __init__(self, in_features, latent_dim=10, hidden_dims=None):
        super(NetVAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.encoder, self.decoder = [], []
        data_features = in_features

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                    nn.ReLU(),
                )
            )
            in_features = h_dim

        self.encoder = nn.Sequential(*modules)


        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        hidden_dims.reverse()
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0])
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_dims[-1], data_features)

    def encode(self, x):
        result = self.encoder(x)

        return self.fc_mean(result), self.fc_log_var(result)

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return torch.sigmoid(result)

    def reparametrization(self, mean, log_var):
        std = 0.5 * torch.exp(log_var)
        z = torch.randn(std.size()) * std + mean

        return z

    def forward(self, x) -> (Tensor, Tensor, Tensor):
        mean, log_var = self.encode(x)
        z = self.reparametrization(mean, log_var)

        return self.decode(z), mean, log_var
