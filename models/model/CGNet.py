from encoder import *
from decoder import *


class CGNet(nn.Module):
    """
    Modello fatto da Matteo Kolyszko
    """
    def __init__(self):
        super(CGNet, self).__init__()
        self.encoder_rgb = RGBEncoder()
        self.encoder_hsi = HSIEncoder()
        self.decoder = Decoder()

    def forward(self, lr_hsi, hr_rgb):
        rgb_feats = self.encoder_rgb(hr_rgb)
        hsi_feats = self.encoder_hsi(lr_hsi)
        x = self.decoder(hsi_feats, rgb_feats)
        return x
