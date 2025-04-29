#from models.SPAN import *
from models.SSPSR import SSPSR
from models.EDSR import EDSR
from models.RCAN import RCAN
from engine import *
from dataset import *
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau


parser = argparse.ArgumentParser(description='Color-Guided Image Super Resolution')
parser.add_argument('--model', type=str, default='1', help='model id')
parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained model')
parser.add_argument('--upscale', type=int, default=4, help='increase upscale factor')
parser.add_argument('--model_path', type=str, default='', help="path to pretrained model")
parser.add_argument('--t_data_path', type=str, default='', help='Train Dataset path')
parser.add_argument('--v_data_path', type=str, default='', help='Val Dataset path')
parser.add_argument('--batch_size', type=int, default='2', help='Training batch size')
parser.add_argument("--epochs", type=int, default=600, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.001")
parser.add_argument("--loss", type=str, default='1', help="loss, default=L1")
parser.add_argument('--save_path', type=str, default='', help="Path to model checkpoint")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run the script on: 'cuda' or 'cpu'. ")

def main():
    model = None
    loss = None
    opt = parser.parse_args()
    print(opt)
    if opt.upscale == 4:
        train_x = os.path.join(opt.t_data_path, 'train_arad1k_x4')
        val_x = os.path.join(opt.v_data_path, 'val_arad1k_x4')
    else:
        train_x = os.path.join(opt.t_data_path, 'train_rad1k_x6')
        val_x = os.path.join(opt.v_data_path, 'val_rad1k_x6')

    train_x_rgb = os.path.join(opt.t_data_path, 'Train_RGB')
    train_y = os.path.join(opt.t_data_path, 'train_arad1k_original')

    val_x_rgb = os.path.join(opt.v_data_path, 'Valid_RGB')
    val_y = os.path.join(opt.v_data_path, 'val_arad1k_original')

    print("===> Loading data")
    train_set = AradDataset(train_x, train_x_rgb, train_y)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

    valid_set = AradDataset(val_x, val_x_rgb, val_y)
    valid_loader = DataLoader(valid_set, batch_size=opt.batch_size, shuffle=False)

    print("===> Building model")
    if opt.model == '1':
        model = EDSR(scale=opt.upscale)
    if opt.model == '2':
        model = RCAN(scale=opt.upscale)
    if opt.model == '3':
        model = SSPSR(n_subs=8, n_ovls=2, n_colors=31, n_blocks=3, n_feats=256, n_scale=opt.upscale, res_scale=0.1)

    model = model.to(opt.device)
    if opt.loss == '1':
        loss = L1Loss()
    elif opt.loss == '2':
        loss = SmoothL1Loss()
    elif opt.loss == '3':
        loss = HybridLoss(spatial_tv=True, spectral_tv=True)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)

    print("===> Setting Scheduler")
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    print("===> Starting Training")
    train(train_loader,
          valid_loader,
          model,
          opt.epochs,
          optimizer,
          opt.device,
          opt.save_path,
          loss,
          scheduler=scheduler)


if __name__ == "__main__":
    main()

