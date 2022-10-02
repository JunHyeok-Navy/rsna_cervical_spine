import loss
import model
import config
import dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data for training
df = pd.read_csv('C:/Users/user/Desktop/deep-learning/cervical/cervical_repo/2d_train_data.csv')

# Model
eff_model = model.EffnetModel()

# Augmentation
train_transform = A.Compose([
    A.Resize(1, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.0625, 
        scale_limit=0.1, 
        rotate_limit=10, 
        p=0.5
    ),
    A.RandomBrightnessContrast(p=0.5),
    ToTensorV2()
    ])
valid_transform = A.Compose([
    A.Resize(1, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
    ToTensorV2()
    ])

## Optimiser
optimizer = optim.Adam(params=eff_model.parameters(), lr=config.LEARNING_RATE)

## Learning rate scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.N_EPOCHS)

# Loop over Folds

final_loss_hist = []
final_val_loss_hist = []
best_val_loss_list = []

for fold in range(config.N_FOLDS):
    # DataFrame
    df_train = df[df['split'] != fold].reset_index(drop=True)
    df_valid = df[df['split'] == fold].reset_index(drop=True)

    # DataLoader
    train_data = dataset.RSNADataset(df_train, config.TRAIN_IMAGE_PATH, transforms=train_transform)
    valid_data = dataset.RSNADataset(df_valid, config.TRAIN_IMAGE_PATH, transforms=valid_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)

    print(f'--------- Fold {fold} Training Start!! ---------')

    loss_hist = []
    val_loss_hist = []
    patience_counter = 0
    best_val_loss = np.inf

    # Loop over epochs
    for epoch in range(config.N_EPOCHS):
        loss_acc = 0
        val_loss_acc = 0
        train_count = 0
        valid_count = 0
        
        # Loop over batches
        with tqdm(train_loader, unit="batch") as tepoch:
            for imgs, y_frac, y_vert in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                # Send to device
                imgs = imgs.to(DEVICE, dtype=torch.float32)
                y_frac = y_frac.to(DEVICE, dtype=torch.float32)
                y_vert = y_vert.to(DEVICE, dtype=torch.float32)

                # Forward pass
                y_frac_pred, y_vert_pred = eff_model(imgs)
                frac_loss = loss.weighted_loss(y_frac_pred, y_frac, reduction='None', verbose=False)
                vert_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert)
                L = config.FRAC_LOSS_WEIGHT * frac_loss + vert_loss

                # Backprop
                L.backward()

                # Update parameters
                optimizer.step()

                # Zero gradients
                optimizer.zero_grad()

                # Track loss
                loss_acc += frac_loss.detach().item()
                train_count += 1
                tepoch.set_postfix(loss=loss_acc/train_count)
        
        # Update learning rate
        scheduler.step()
        
        # Don't update weights
        with torch.no_grad():
            # Validate
            for val_imgs, val_y_frac, val_y_vert in valid_loader:
                # Reshape
                val_imgs = val_imgs.to(DEVICE, dtype=torch.float32)
                val_y_frac = val_y_frac.to(DEVICE, dtype=torch.float32)
                val_y_vert = val_y_vert.to(DEVICE, dtype=torch.float32)

                # Forward pass
                val_frac_pred, val_vert_pred = eff_model(val_imgs)
                val_frac_loss = loss.weighted_loss(val_frac_pred, val_y_frac, reduction='None', verbose=False)
                val_vert_loss = torch.nn.functional.binary_cross_entropy_with_logits(val_y_vert, val_y_vert)
                val_loss = config.FRAC_LOSS_WEIGHT * val_frac_loss + val_vert_loss

                # Track loss
                val_loss_acc += val_frac_loss.item()
                valid_count += 1
            print(f'Val Loss : {val_loss_acc / valid_count}')
        
        # Save loss history
        loss_hist.append(loss_acc/train_count)
        val_loss_hist.append(val_loss_acc/valid_count)
        
        # Print loss
        if (epoch+1)%1==0:
            print(f'Epoch {epoch+1}/{config.N_EPOCHS}, loss {loss_acc/train_count:.5f}')
        
        # Save model (& early stopping)
        if (val_loss_acc/valid_count) < best_val_loss:
            best_val_loss = val_loss_acc/valid_count
            patience_counter=0
            print('Valid loss improved --> saving model')
            torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': eff_model.state_dict(),
                        'optimiser_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss_acc/train_count,
                        'val_loss': val_loss_acc/valid_count,
                        }, f"{config.MODEL_NAME}_fold_{fold}.pt")
        else:
            patience_counter+=1
            
            if patience_counter==config.PATIENCE:
                break

    final_loss_hist.append(loss_hist)
    final_val_loss_hist.append(val_loss_hist)
    best_val_loss_list.append(best_val_loss)

print('')
print('Training complete!')