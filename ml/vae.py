import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def training_step(self, batch, batch_idx):

        x = batch['feature']
        if (self.global_step == 1):
            #  add computation graph
            self.logger.experiment.add_graph(self, x)

        recon_x, mu, logvar = self(x)
        loss = self.loss_function(recon_x, x, mu, logvar)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
       #  the function is called after every epoch is completed

       # calculating average loss  
       avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

       # calculating correect and total predictions
       #correct=sum([x["correct"] for  x in outputs])
       #total=sum([x["total"] for  x in outputs])

       # logging using tensorboard logger
       self.logger.experiment.add_scalar("Loss/Train",
                                           avg_loss,
                                           self.current_epoch)
       
       #self.logger.experiment.add_scalar("Accuracy/Train",
                                           #correct/total,
                                           #self.current_epoch)

       epoch_dictionary={
           # required
           'loss': avg_loss}

       return epoch_dictionary

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.01)


class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # layer 1
            nn.Linear(
                in_features=75,
                out_features=512
            ),
            nn.ReLU(),
            # layer 2
            nn.Linear(
                in_features=512,
                out_features=512
            ),
            nn.ReLU(),
            # layer 3
            nn.Linear(
                in_features=512,
                out_features=1024
            ),
            nn.ReLU(),
        )

        # output
        self.mu = nn.Linear(
            in_features=1024,
            out_features=100
        )
        self.logvar = nn.Linear(
            in_features=1024,
            out_features=100
        )

    def forward(self, x):
        h = self.fc(x)
        return self.mu(h), self.logvar(h)


class Decoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # layer 1
            nn.Linear(
                in_features=100,
                out_features=1024
            ),
            nn.ReLU(),
            # layer 2
            nn.Linear(
                in_features=1024,
                out_features=512
            ),
            nn.ReLU(),
            # layer 3
            nn.Linear(
                in_features=512,
                out_features=512
            ),
            nn.ReLU(),
            # output
            nn.Linear(
                in_features=512,
                out_features=75
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)
