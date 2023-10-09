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
from data_loading.datasets import InaturalistEmbedding
from data_loading.core import NShotTaskSampler
from utilites.utils import pairwise_distances

from config import ROOT_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=5, type=int)  # n shot task
parser.add_argument('--n-test', default=5, type=int)
parser.add_argument('--k-train', default=650, type=int)
parser.add_argument('--k-test', default=650, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=5, type=int)
parser.add_argument('--episodes_per_epoch', default=100, type=int)
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--drop_lr_every', default=40, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--numexp', default=1, type=int)
args = parser.parse_args()


class MobileNEtLinearHead(nn.Module):
    def __init__(self):
        super(MobileNEtLinearHead, self).__init__()
        # self.base = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384',
        #                               pretrained=True)  # Huge = 2816 , Large = 1536
        # self.base.head.fc = nn.Identity()
        # for param in self.base.parameters():
        #     param.requires_grad = False

        self.base = timm.create_model('mobilenetv3_large_100.miil_in21k', pretrained=True)
        self.base.classifier = nn.Identity()

        self.head = nn.Sequential(nn.Linear(OUTPUT_DIM + NO_CLUSTER, OUTPUT_DIM),
                                  nn.GELU(),
                                  nn.Linear(OUTPUT_DIM, OUTPUT_DIM))

    def forward(self, x, prob):
        x = self.base(x)
        model_input = torch.cat((x, prob), dim=1)
        y = self.head(model_input)
        return y


NO_CLUSTER = 74
OUTPUT_DIM = 512
SCHEDULER_STEP = 10

param_string = f'CLIP-VITB32-224px-TwoLinear{OUTPUT_DIM}-on74SC-OneHotConditionTraining-on-InatAnimalia-{args.k_train}way-{args.n_train}shot_support-{args.q_train}shot_query-lr-{args.lr}'
print(param_string)

episodes_per_epoch = args.episodes_per_epoch
n_epochs = args.n_epochs
drop_lr_every = args.drop_lr_every

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

background = InaturalistEmbedding('background', model_family='clip', model_name='vitb32')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
    num_workers=4
)
print(f'len background = {len(background)}, |classes| = {background.num_classes()}')
evaluation = InaturalistEmbedding('evaluation', model_family='clip', model_name='vitb32')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    num_workers=4
)
print(f'len evaluation = {len(evaluation)}, |classes| = {evaluation.num_classes()}')
lin_model = nn.Sequential(nn.Linear(OUTPUT_DIM + NO_CLUSTER, OUTPUT_DIM),
                          nn.GELU(),
                          nn.Linear(OUTPUT_DIM, OUTPUT_DIM))
# lin_model = MobileNEtLinearHead()
# for param in model.parameters():
#     param.requires_grad = False
#
# model.to(device)
lin_model.to(device, dtype=torch.float32)

optimiser = torch.optim.Adam(lin_model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimiser, mode='max', factor=0.1, patience=4)

criterion = nn.CrossEntropyLoss()
writer = SummaryWriter(f'runs/{param_string}')
best_acc = float('-inf')
for epoch in tqdm(range(1, args.n_epochs + 1)):
    running_loss = 0.0
    running_acc = 0.0
    for batch_index, batch in enumerate(background_taskloader):
        x, y_true = batch
        x = x.to(device, dtype=torch.float32)
        # Create dummy 0-(num_classes - 1) label
        y = torch.arange(0, args.k_train, 1 / args.q_train).long().to(
            device)  # (q_train * k_train,) pseudo labels for queries

        condition = []
        for label in y_true:
            super_class_id = background.superclass_index(label.item())
            sc_tensor = torch.tensor(super_class_id)
            sample_condition = nn.functional.one_hot(sc_tensor, NO_CLUSTER)
            condition.append(sample_condition)

        condition = torch.stack(condition)
        condition = condition.to(device)

        # Zero gradients
        # model.eval()
        lin_model.train()
        optimiser.zero_grad()

        # Embed all samples
        # with torch.no_grad():
        #     embeddings = model(x)
        #     lin_input = embeddings.detach()

        # embeddings = lin_model(lin_input)
        model_input = torch.cat((x, condition), dim=1)
        embeddings = lin_model(model_input)

        # embeddings = lin_model(x, condition)

        # Samples are ordered by the NShotWrapper class as follows:
        support = embeddings[:args.n_train * args.k_train]
        queries = embeddings[args.n_train * args.k_train:]

        prototypes = support.reshape(args.k_train, args.n_train, -1).mean(dim=1)

        # Calculate squared distances between all queries and all prototypes
        # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
        distances = pairwise_distances(queries, prototypes, 'l2')  # (q_train, k_train)

        loss = criterion(-distances, y)
        running_loss += loss.detach().item()
        # Prediction probabilities are softmax over distances
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
            x, y_true = batch
            x = x.to(device, dtype=torch.float32)
            # Create dummy 0-(num_classes - 1) label
            y = torch.arange(0, args.k_test, 1 / args.q_test).long().to(device)

            condition = []
            for label in y_true:
                super_class_id = background.superclass_index(label.item())
                sc_tensor = torch.tensor(super_class_id)
                sample_condition = nn.functional.one_hot(sc_tensor, NO_CLUSTER)
                condition.append(sample_condition)

            condition = torch.stack(condition)
            condition = condition.to(device)

            # with torch.no_grad():
            #     embeddings = model(x)
            #     lin_input = embeddings.detach()

            # embeddings = lin_model(lin_input)
            model_input = torch.cat((x, condition), dim=1)
            embeddings = lin_model(model_input)

            # embeddings = lin_model(x, condition)

            support = embeddings[:args.n_test * args.k_test]
            queries = embeddings[args.n_test * args.k_test:]
            prototypes = support.reshape(args.k_test, args.n_test, -1).mean(dim=1)

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
