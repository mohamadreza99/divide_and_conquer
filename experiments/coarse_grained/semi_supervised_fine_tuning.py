import sys

import timm
import torch
import torch.nn as nn

import sys

import torch
import torch.nn as nn

sys.path.append("../..")
from data_loading.datasets import InaturalistPlantaeEmbedding, InaturalistEmbedding

from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs --bind_all
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import ROOT_PATH

# dino V2
# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14') 1024
# dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14') 1536

MIXED_TUNE = False
NUM_OF_SUPERCLASSES = 74
LR = 1e-3
EPOCHS = 80
BATCH_SIZE = 64
DATASET_OUTPUT_DIM = 1536

param_string = f'phi-dino-vitg14-FC{NUM_OF_SUPERCLASSES}D-finetuned-on_pseudoLabels-MIXED_TUNE={MIXED_TUNE}-Animalia'
print(param_string)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device.type == 'cuda'

# model.to(device)
# for param in model.parameters():
#     param.requires_grad = False
linmodel = nn.Linear(DATASET_OUTPUT_DIM, NUM_OF_SUPERCLASSES)

load_path = ROOT_PATH + f'/experiments/dino_12/persistents/phi-dino-vitg14-Fully66D-LRSch-on-Animalia.pth'
print(linmodel.load_state_dict(torch.load(load_path, map_location=device)))
linmodel.to(device)

background = InaturalistEmbedding('evaluation', model_family='dino', model_name='vitg14', coarse_label=True)
# background = Cars('evaluation', coarse_label=True)
# maintain 5 shot of each class in evaluation and pseudo label them to fine tune the model
queries = []
for cls in sorted(background.df.class_id.unique()):
    tmpdf = background.df[background.df['class_id'] == cls]
    queries.extend(list(tmpdf.index[5:]))
background.df = background.df.drop(queries).reset_index(drop=True)
# attach pseudo labels
pseudo_label_acc = 0.0
with torch.no_grad():
    for cls_id in tqdm(sorted(background.df['class_id'].unique())):
        cls_df = background.df[background.df['class_id'] == cls_id]
        image_list = []
        for indx, row in cls_df.iterrows():
            image, label = background[indx]
            image = image.to(device, dtype=torch.float32).unsqueeze(0)
            emb = image
            image_list.append(image)
        phi_input = torch.concat(image_list)
        super_class_logit = linmodel(phi_input)
        logit = super_class_logit.log_softmax(dim=-1).sum(dim=0)
        pseudo_label = torch.argmax(logit).item()
        if pseudo_label == cls_df.iloc[0,4]:
            pseudo_label_acc += 1
        background.df.loc[cls_df.index, 'super_class_id'] = pseudo_label
print(f'pseudo label acc: {pseudo_label_acc/background.num_classes()}')

if MIXED_TUNE:
    trainDS = InaturalistEmbedding('background', model_family='dino', model_name='vitg14', coarse_label=True)
    background.df = background.df.append(trainDS.df, ignore_index=True)

background.df = background.df.assign(id=background.df.index.values)

background_taskloader = DataLoader(
    background,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=4
)

evaluation = InaturalistEmbedding('evaluation',model_family='dino', model_name='vitg14', coarse_label=True)
linmodel.eval()
with torch.no_grad():
    for cls_id in tqdm(sorted(evaluation.df['class_id'].unique())):
        cls_df = evaluation.df[evaluation.df['class_id'] == cls_id]
        image_list = []
        for indx, row in cls_df[:5].iterrows():
            image, label = evaluation[indx]
            image = image.to(device, dtype=torch.float32).unsqueeze(0)
            emb = image
            image_list.append(image)
        phi_input = torch.concat(image_list)
        super_class_logit = linmodel(phi_input)
        logit = super_class_logit.log_softmax(dim=-1).sum(dim=0)
        pseudo_label = torch.argmax(logit).item()
        evaluation.df.loc[cls_df.index[5:], 'super_class_id'] = pseudo_label
queries = []
for cls in sorted(evaluation.df.class_id.unique()):
    tmpdf = evaluation.df[evaluation.df['class_id'] == cls]
    queries.extend(list(tmpdf.index[:5]))
evaluation.df = evaluation.df.drop(queries).reset_index(drop=True)
evaluation.df = evaluation.df.assign(id=evaluation.df.index.values)
evaluation_taskloader = DataLoader(
    evaluation,
    batch_size=BATCH_SIZE,
    num_workers=4
)

optimiser = torch.optim.Adam(linmodel.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'max', verbose=True, min_lr=1e-6)
criterion = nn.CrossEntropyLoss()
best_acc = float('-inf')
writer = SummaryWriter(f'runs/{param_string}')
for epoch in tqdm(range(1, EPOCHS + 1)):
    running_loss = 0.0
    running_acc = 0.0
    linmodel.train()
    for batch_index, batch in enumerate(background_taskloader):
        x, y = batch

        x = x.to(device)
        y = y.to(device)

        # Zero gradients
        optimiser.zero_grad()

        # Embed all samples
        yhat = linmodel(x)

        loss = criterion(yhat, y)
        running_loss += loss.detach().item() * len(y)

        y_pred = torch.argmax(yhat, dim=1)
        running_acc += torch.eq(y_pred, y).sum().item()

        loss.backward()
        optimiser.step()
    writer.add_scalar('train_loss', running_loss / len(background), epoch)
    writer.add_scalar('train_acc', running_acc / len(background), epoch)

    # evaluation
    running_loss = 0.0
    running_acc = 0.0
    top5_acc = 0.0
    linmodel.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(evaluation_taskloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            # Embed all samples
            yhat = linmodel(x)

            loss = criterion(yhat, y)
            running_loss += loss.detach().item() * len(y)
            y_pred = torch.argmax(yhat, dim=1)  # (q_test * k_test,)
            running_acc += torch.eq(y_pred, y).sum().item()
            y_pred5 = yhat.topk(k=5, dim=1)[1]
            y_pred5 = y_pred5.flatten()
            y = y.repeat_interleave(5, dim=0)
            top5_acc += torch.eq(y_pred5, y).sum().item()

        writer.add_scalar('eval_loss', running_loss / len(evaluation), epoch)
        writer.add_scalar('eval_acc', running_acc / len(evaluation), epoch)

        if running_acc > best_acc:
            print(f'saving model ..., best_acc = {running_acc / len(evaluation)}, top5_acc = {top5_acc / len(evaluation)}')
            best_acc = running_acc
            torch.save(linmodel.state_dict(),
                       ROOT_PATH + f'/experiments/dino_12/persistents/{param_string}.pth')

    scheduler.step(running_acc)
