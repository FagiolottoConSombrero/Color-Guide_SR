from tqdm import tqdm
import torch
from utils import *

def training_step(train_loader, model, optimizer, device, criterion):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc="Training", leave=True, dynamic_ncols=True)

    for batch, (hsi, rgb, y) in enumerate(loop):
        # Send data to GPU
        hsi, rgb, y = hsi.to(device), rgb.to(device), y.to(device)

        # Forward Pass
        y_pred = model(hsi, rgb)

        # Calculate loss
        loss_1 = criterion(y_pred, y)
        total_loss += loss_1.item()

        # Optimizer reset step
        optimizer.zero_grad()

        # Loss Backpropagation
        loss_1.backward()

        # Optimizer step
        optimizer.step()

    total_loss /= len(train_loader)
    return total_loss


def validation_step(val_loader, model, device, criterion):
    model.eval()
    total_loss = 0
    loop = tqdm(val_loader, desc="Validation", leave=True)

    with torch.inference_mode():
        for batch, (hsi, rgb, y) in enumerate(loop):
            hsi, rgb,y = hsi.to(device), rgb.to(device), y.to(device)
            y_pred = model(hsi, rgb)

            # Calculate structural similarity index loss
            loss_1 = criterion(y_pred, y)
            total_loss += loss_1.item()

    total_loss /= len(val_loader)

    return total_loss


def train(train_loader,
          val_loader,
          model,
          epochs,
          optimizer,
          device,
          best_model_path,
          criterion,
          patience=50,
          scheduler=None):  # <--- aggiunto

    early_stopping = EarlyStopping(patience=patience, mode='min')

    for epoch in tqdm(range(epochs), desc="All"):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = training_step(train_loader, model, optimizer, device, criterion)
        val_loss = validation_step(val_loader, model, device, criterion)

        # Aggiorna lo scheduler solo se è ReduceLROnPlateau
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Stampa il learning rate corrente
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"LR: {current_lr:.6f} | "
            f"Train loss: {train_loss:.8f} | "
            f"Val loss: {val_loss:.8f}"
        )
        print("-------------")

        if check_early_stopping(val_loss, model, early_stopping, epoch, best_model_path):
            break

    # Ripristina i pesi migliori
    model.load_state_dict(torch.load(best_model_path))
    print(f"Restored best model weights with val_loss: {early_stopping.best_val:.6f}")