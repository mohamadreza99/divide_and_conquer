import sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm

sys.path.append("../..")
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs --bind_all
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from data_loading.datasets import InaturalistPlantae
from data_loading.core import NShotTaskSampler
from utilites.utils import pairwise_distances

from config import ROOT_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=5, type=int)  # n shot task
parser.add_argument('--n-test', default=5, type=int)
parser.add_argument('--k-train', default=100, type=int)
parser.add_argument('--k-test', default=100, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=5, type=int)
parser.add_argument('--episodes_per_epoch', default=100, type=int)
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--drop_lr_every', default=40, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--numexp', default=1, type=int)
args = parser.parse_args()

OUTPUT_DIM = 1536

param_string = f'new-prototypical-baseline-ConNextV2-224px-TwoLinear{OUTPUT_DIM}-66SC-on-Cars-{args.k_train}way-{args.n_train}shot_support-{args.q_train}shot_query-lr-{args.lr}'
print(param_string)

episodes_per_epoch = args.episodes_per_epoch
n_epochs = args.n_epochs
drop_lr_every = args.drop_lr_every

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# background = InaturalistPlantaeEmbedding('background', model_name='vitg14')
background = InaturalistPlantae('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
    num_workers=4
)
print(f'len background = {len(background)}, |classes| = {background.num_classes()}')
# evaluation = InaturalistPlantaeEmbedding('evaluation', model_name='vitg14')
evaluation = InaturalistPlantae('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    num_workers=4
)
print(f'len evaluation = {len(evaluation)}, |classes| = {evaluation.num_classes()}')

# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')  # testing on VIT Large
# lin_model = DinoLinearHead(1024, OUTPUT_DIM)
# lin_model = timm.create_model('mobilenetv3_large_100.miil_in21k', pretrained=True)

lin_model = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True)  # Huge = 2816 , Large = 1536

for param in lin_model.parameters():
    param.requires_grad = False

lin_model.head.fc = nn.Sequential(nn.Linear(OUTPUT_DIM, OUTPUT_DIM),
                                  nn.GELU(),
                                  nn.Linear(OUTPUT_DIM, OUTPUT_DIM))

# lin_model.eval()
#
# lin_model = nn.Sequential(nn.Linear(OUTPUT_DIM, OUTPUT_DIM),
#                           nn.GELU(),
#                           nn.Linear(OUTPUT_DIM, OUTPUT_DIM))

lin_model.to(device, dtype=torch.float32)

optimiser = torch.optim.Adam(lin_model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimiser, mode='max', factor=0.1, patience=7, verbose=True)

criterion = nn.CrossEntropyLoss()
writer = SummaryWriter(f'runs/{param_string}')
best_acc = float('-inf')
for epoch in tqdm(range(1, args.n_epochs + 1)):
    running_loss = 0.0
    running_acc = 0.0
    for batch_index, batch in enumerate(background_taskloader):
        x, _ = batch
        x = x.float().to(device)  # (540, 3, 84, 84)
        # Create dummy 0-(num_classes - 1) label
        y = torch.arange(0, args.k_train, 1 / args.q_train).long().to(
            device)  # (q_train * k_train,) pseudo labels for queries

        # Zero gradients
        # model.eval()
        lin_model.train()
        optimiser.zero_grad()

        # Embed all samples
        # with torch.no_grad():
        #     embeddings = model(x)
        #     lin_input = embeddings.detach()

        # embeddings = lin_model(lin_input)
        embeddings = lin_model(x)

        # Samples are ordered by the NShotWrapper class as follows:
        # k lots of n support samples from a particular class
        # k lots of q query samples from those classes
        support = embeddings[:args.n_train * args.k_train]
        queries = embeddings[args.n_train * args.k_train:]

        prototypes = support.reshape(args.k_train, args.n_train, -1).mean(dim=1)

        # Calculate squared distances between all queries and all prototypes
        # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
        distances = pairwise_distances(queries, prototypes, 'l2')  # (q_train, k_train)

        loss = criterion(-distances, y)
        running_loss += loss.detach().item()
        y_pred = torch.argmax(-distances, dim=1)  # (300,)
        running_acc += torch.eq(y_pred, y).sum().item() / y_pred.shape[0]

        loss.backward()
        optimiser.step()

    writer.add_scalar('train_loss', running_loss / len(background_taskloader), epoch)
    writer.add_scalar('train_acc', running_acc / len(background_taskloader), epoch)

    # evaluation
    running_loss = 0.0
    running_acc = 0.0
    # model.eval()
    lin_model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(evaluation_taskloader):
            x, true_labels = batch
            x = x.float().to(device)
            # Create dummy 0-(num_classes - 1) label
            y = torch.arange(0, args.k_test, 1 / args.q_test).long().to(device)  # (300,)

            # with torch.no_grad():
            #     embeddings = model(x)
            #     lin_input = embeddings.detach()

            # embeddings = lin_model(lin_input)
            embeddings = lin_model(x)

            support = embeddings[:args.n_test * args.k_test]
            queries = embeddings[args.n_test * args.k_test:]
            prototypes = support.reshape(args.k_test, args.n_test, -1).mean(dim=1)

            # Calculate squared distances between all queries and all prototypes
            distances = pairwise_distances(queries, prototypes, 'l2')  # (q_train, k_train)

            loss = criterion(-distances, y)
            running_loss += loss.detach().item()
            y_pred = torch.argmax(-distances, dim=1)  # (q_test * k_test,)
            running_acc += torch.eq(y_pred, y).sum().item() / y_pred.shape[0]

        writer.add_scalar('eval_loss', running_loss / len(evaluation_taskloader), epoch)
        writer.add_scalar('eval_acc', running_acc / len(evaluation_taskloader), epoch)

        scheduler.step(running_acc)

        if running_acc > best_acc:
            print(f'saving model ..., best_acc = {running_acc / len(evaluation_taskloader)}')
            best_acc = running_acc
            torch.save(lin_model.state_dict(),
                       ROOT_PATH + f'/experiments/persistents/{param_string}.pth')

writer.close()
