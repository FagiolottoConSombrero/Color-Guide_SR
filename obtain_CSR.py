from engine import *
from models.mst import *
import argparse
from dataset import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


parser = argparse.ArgumentParser(description='Color-Guided Image Super Resolution')
parser.add_argument('--t_data_path', type=str, default='', help='Train Dataset path')
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run the script on: 'cuda' or 'cpu'. ")
parser.add_argument('--batch_size', type=int, default='16', help='Training batch size')
parser.add_argument('--save_path', type=str, default='', help="Path to model checkpoint")


class CSRLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.conv.weight, 1.0 / in_channels)

    def forward(self, x):
        return torch.clamp(self.conv(x), min=0.0)

    def clamp_weights(self):
        with torch.no_grad():
            self.conv.weight.data.clamp_(min=0.0)


def smoothness_loss(weight):
    w = weight.view(3, -1)  # (3, B)
    diff = w[:, 1:] - w[:, :-1]
    return torch.mean(diff ** 2)


def total_loss(x_hat, x_gt, weight, eta3=0.01, eta4=0.1):
    mse = F.mse_loss(x_hat, x_gt)
    l2 = torch.mean(weight ** 2)
    smooth = smoothness_loss(weight)
    return mse + eta3 * l2 + eta4 * smooth


def train_csr_autoencoder(csr_layer, recon_net, train_loader, val_loader, optimizer,
                          device='cuda', num_epochs=100, patience=50,
                          best_model_path='best_csr.pth', scheduler=None):
    csr_layer.to(device)
    recon_net.to(device)
    csr_layer.train()
    recon_net.eval()

    early_stopping = EarlyStopping(patience=patience, mode='min')

    for epoch in tqdm(range(num_epochs), desc="All"):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        total_train = 0.0
        for x_hsi in train_loader:
            x_hsi = x_hsi.to(device)
            rgb_sim = csr_layer(x_hsi)
            x_hat = recon_net(rgb_sim)
            loss = total_loss(x_hat, x_hsi, csr_layer.conv.weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            csr_layer.clamp_weights()
            total_train += loss.item()

        total_val = 0.0
        with torch.no_grad():
            for x_val in val_loader:
                x_val = x_val.to(device)
                rgb_sim_val = csr_layer(x_val)
                x_hat_val = recon_net(rgb_sim_val)
                val_loss = total_loss(x_hat_val, x_val, csr_layer.conv.weight)
                total_val += val_loss.item()

        avg_train_loss = total_train / len(train_loader)
        avg_val_loss = total_val / len(val_loader)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"LR: {current_lr:.6f} | Train loss: {avg_train_loss:.6f} | Val loss: {avg_val_loss:.6f}")
        print("-------------")

        if check_early_stopping(avg_val_loss, model, early_stopping, epoch, best_model_path):
            break

    csr_layer.load_state_dict(torch.load(best_model_path))
    print(f"Restored best CSR weights with val_loss: {early_stopping.best_val:.6f}")
    return csr_layer.conv.weight.detach().cpu()


opt = parser.parse_args()
model = MST_Plus_Plus()
checkpoint = torch.load('/home/ubuntu/HSI-RGB-SuperResolution/model_weights/mst_plus_plus.pth',
                        map_location=opt.device)
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)


train_x = os.path.join(opt.t_data_path, 'train_arad1k_x4')
val_x = os.path.join(opt.v_data_path, 'val_arad1k_x4')

train_x_rgb = os.path.join(opt.t_data_path, 'Train_RGB')
train_y = os.path.join(opt.t_data_path, 'train_arad1k_original')

val_x_rgb = os.path.join(opt.v_data_path, 'Valid_RGB')
val_y = os.path.join(opt.v_data_path, 'val_arad1k_original')

print("===> Loading data")
train_set = AradDataset(train_x, train_x_rgb, train_y)
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

valid_set = AradDataset(val_x, val_x_rgb, val_y)
valid_loader = DataLoader(valid_set, batch_size=opt.batch_size, shuffle=False)

csr_layer = CSRLayer(in_channels=31)
optimizer = torch.optim.Adam(list(csr_layer.parameters()) + list(model.parameters()), lr=0.001)

print("===> Setting Scheduler")
scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )
learned_csr = train_csr_autoencoder(csr_layer, model, train_loader, valid_loader, optimizer,
                                    best_model_path=opt.save_path, scheduler=scheduler)
torch.save(learned_csr, 'csr_optimized_from_autoencoder.pt')
