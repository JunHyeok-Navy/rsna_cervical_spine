import os
import loss
import timm
import model
import torch
import config
import dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

valid_transform = A.Compose([
    A.Resize(*config.IMAGE_SIZE),
    ToTensorV2()
    ])

def load_model(model, path):
    data = torch.load(path, map_location=DEVICE)
    model.load_state_dict(data, strict=False)
    return model

def evaluate_effnet(model, ds, max_batches=config.PREDICT_MAX_BATCHES, shuffle=False):
    torch.manual_seed(42)
    model = model.to(DEVICE)
    dl_test = torch.utils.data.DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    pred_frac = []
    pred_vert = []
    with torch.no_grad():
        model.eval()
        frac_losses = []
        vert_losses = []
        with tqdm(dl_test, desc='Eval', miniters=10) as progress:
            for i, (X, y_frac, y_vert) in enumerate(progress):
                X = X.to(DEVICE, dtype=torch.float32)
                y_frac = y_frac.to(DEVICE, dtype=torch.float32)
                y_vert = y_vert.to(DEVICE, dtype=torch.float32)

                y_frac_pred, y_vert_pred = model.forward(X)
                frac_loss = loss.weighted_loss(y_frac_pred, y_frac).item()
                vert_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert).item()
                pred_frac.append(torch.sigmoid(y_frac_pred))
                pred_vert.append(torch.sigmoid(y_vert_pred))
                frac_losses.append(frac_loss)
                vert_losses.append(vert_loss)

                if i >= max_batches:
                    break
        return np.mean(frac_losses), np.mean(vert_losses), torch.cat(pred_frac).cpu().numpy(), torch.cat(pred_vert).cpu().numpy()

def gen_effnet_predictions(model, df_train):
    df_train_predictions = []
    ds_eval = dataset.RSNADataset(df_train.query('split == 0'), config.TRAIN_IMAGE_PATH, valid_transform)
    frac_loss, vert_loss, effnet_pred_frac, effnet_pred_vert = evaluate_effnet(model, ds_eval, config.PREDICT_MAX_BATCHES)
    df_effnet_pred = pd.DataFrame(data=np.concatenate([effnet_pred_frac, effnet_pred_vert], axis=1),
                                    columns=[f'C{i}_effnet_frac' for i in range(1, 8)] +
                                            [f'C{i}_effnet_vert' for i in range(1, 8)])

    df = pd.concat(
        [df_train.query('split == 0').head(len(df_effnet_pred)).reset_index(drop=True), df_effnet_pred],
        axis=1
    ).sort_values(['StudyInstanceUID', 'Slice'])
    df_train_predictions.append(df)
    df_train_predictions = pd.concat(df_train_predictions)
    return df_train_predictions

def main():
    # Model
    backbone = timm.create_model(config.MODEL_NAME, pretrained=True, in_chans=1)
    eff_model = model.EffnetModel(model=backbone)
    eff_model.to(DEVICE)

    saved_model = load_model(eff_model, "C:/Users/user/Desktop/deep-learning/cervical/cervical_repo/densenet121_fold_0.pt")

    df_train_predictions = gen_effnet_predictions(saved_model, pd.read_csv('C:/Users/user/Desktop/deep-learning/cervical/cervical_repo/group_shuffled_2d_df.csv'))
    df_train_predictions.to_csv('df_train_predictions.csv', index=False)

if __name__ == '__main__':
    main()