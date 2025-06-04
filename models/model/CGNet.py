from models.model.encoder import *
from models.model.decoder import *



class CGNet(nn.Module):
    """
    Modello fatto da Matteo Kolyszko
    """
    def __init__(self, out_ch=64, scale=4):
        super(CGNet, self).__init__()
        if scale == 4:
            self.encoder_rgb = RGBEncoder(out_ch=out_ch)
            self.encoder_hsi = HSIEncoder(out_ch=out_ch)
            self.decoder = Decoder(in_ch=out_ch)
        elif scale == 6:
            self.encoder_rgb = RGBEncoder6(out_ch=out_ch)
            self.encoder_hsi = HSIEncoder6(out_ch=out_ch)
            self.decoder = Decoder6(in_ch=out_ch)

    def forward(self, lr_hsi, hr_rgb):
        rgb_feats = self.encoder_rgb(hr_rgb)
        hsi_feats = self.encoder_hsi(lr_hsi)
        x = self.decoder(hsi_feats, rgb_feats)
        return x
