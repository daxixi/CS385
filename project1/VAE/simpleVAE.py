import torch
import torch.nn.functional as F
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self,hidden):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3072, 1024), nn.ReLU()
        )
        self.mean = nn.Linear(1024, hidden)
        self.logvar = nn.Linear(1024, hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3072),
            nn.Sigmoid(),
        )

    def sample(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = x.view(x.size(0),3072)
        x = self.encoder(x)
        log_var = self.logvar(x)
        mean = self.mean(x)

        z = self.sample(log_var, mean)
        x = self.decoder(z)

        x=x.view(x.size(0),3,32,32)
        return x, mean, log_var

class CVAE(nn.Module):
    def __init__(self,hidden):
        super(CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3,2,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3,2,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3,2,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3,1),
            nn.LeakyReLU(0.2),
        )
        self.mean = nn.Linear(1024, hidden)
        self.logvar = nn.Linear(1024, hidden)
        self.fc = nn.Linear(hidden, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 4, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2,padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, 2,padding=2),
            nn.Sigmoid(),
        )

    def sample(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0),-1)
        log_var = self.logvar(x)
        mean = self.mean(x)
        z = self.sample(log_var, mean)
        x = self.fc(z)
        x = x.view(x.size(0),1024,1,1)
        x = self.decoder(x)

        return x, mean, log_var
