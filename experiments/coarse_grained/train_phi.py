import sys
import timm
import torch
import torch.nn as nn

sys.path.append("../..")
from torch.utils.tensorboard import SummaryWriter
from data_loading.datasets import InaturalistPlantaeEmbedding, Inaturalist2021mini, InaturalistEmbedding
# tensorboard --logdir=runs --bind_all
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import ROOT_PATH

# dino V2
# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14') #1024
# dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14') #1536

# convnext2
# model = timm.create_model('convnextv2_huge.fcmae', pretrained=True) 2816

# mobileNet
# model = timm.create_model('mobilenetv3_large_100.miil_in21k', pretrained=True)
# model.classifier = nn.Identity()


NUM_OF_SUPERCLASSES = 74
LR = 1e-3
EPOCHS = 80
BATCH_SIZE = 64
DATASET_OUTPUT_DIM = 1536

param_string = f'phi-dino-vitg14-Fully{NUM_OF_SUPERCLASSES}D-LRSch-on-Animalia'
print(param_string)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

background = InaturalistEmbedding('background', model_family='dino', model_name='vitg14', coarse_label=True)
background_taskloader = DataLoader(
    background,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=6
)

evaluation = InaturalistEmbedding('evaluation', model_family='dino', model_name='vitg14', coarse_label=True)
evaluation_taskloader = DataLoader(
    evaluation,
    batch_size=BATCH_SIZE,
    num_workers=6
)
print(f"len of all background images are {len(background)}")
print(f"len of all evaluation images are {len(evaluation)}")

# model.to(device)
# for param in model.parameters():
#     param.requires_grad = False
# linmodel = nn.Sequential(model, nn.Linear(DATASET_OUTPUT_DIM, NUM_OF_SUPERCLASSES))
linmodel = nn.Linear(DATASET_OUTPUT_DIM, NUM_OF_SUPERCLASSES)
linmodel.to(device)

optimiser = torch.optim.Adam(linmodel.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'max', verbose=True, min_lr=1e-6)
criterion = nn.CrossEntropyLoss()
best_acc = float('-inf')
writer = SummaryWriter(f'runs/{param_string}')
for epoch in tqdm(range(1, EPOCHS + 1)):
    running_loss = 0.0
    running_acc = 0.0
    # model.train()
    linmodel.train()
    for batch_index, batch in enumerate(background_taskloader):
        x, y = batch

        x = x.to(device)
        y = y.to(device)

        # Zero gradients
        # model.train()
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
    # model.eval()
    linmodel.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(evaluation_taskloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            # Embed all samples
            # x = model(x)
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