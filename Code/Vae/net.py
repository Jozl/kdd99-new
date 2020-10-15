import torch
from torch import nn, Tensor


class NetVAE(nn.Module):
    def __init__(self, in_channels, latent_dim=2, hidden_dims=[32, 64, 128, 256, 512]):
        super(NetVAE, self).__init__()

        self.encoder, self.decoder = [], []

        in_channels += 1  # To account for the extra label channel
        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, x):
        for fc in self.encoder:
            x = torch.relu(fc(x))

        return self.fc_mean(x), self.fc_log_var(x)

    def decode(self, z):
        for fc in self.decoder[0: -1]:
            z = torch.relu(fc(z))

        return torch.sigmoid(self.final_layer(z))

    def reparametrization(self, mean, log_var):
        std = 0.5 * torch.exp(log_var)
        z = torch.randn(std.size()) * std + mean

        return z

    def forward(self, x) -> (Tensor, Tensor, Tensor):
        mean, log_var = self.encode(x)
        z = self.reparametrization(mean, log_var)

        return self.decode(z), mean, log_var
