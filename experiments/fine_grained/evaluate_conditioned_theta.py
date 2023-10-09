import sys

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append("../..")
from data_loading.datasets import InaturalistEmbeddinga
from utilites.utils import pairwise_distances
from config import ROOT_PATH
import pickle
import timm

THETA_OUTPUT_DIM = 512
NO_CLUSTER = 66
N_TRAIN = 5
K = 5
TOP_ACC = 5

print('k:', K)
print('top acc:', TOP_ACC)


class TimmHead(nn.Module):
    def __init__(self):
        super(TimmHead, self).__init__()
        self.base = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384',
                                      pretrained=True)  # Huge = 2816 , Large = 1536
        self.base.head.fc = nn.Identity()

        # self.base = timm.create_model('mobilenetv3_large_100.miil_in21k', pretrained=True)
        # self.base.classifier = nn.Identity()
        self.head = nn.Sequential(nn.Linear(THETA_OUTPUT_DIM + NO_CLUSTER, THETA_OUTPUT_DIM),
                                  nn.GELU(),
                                  nn.Linear(THETA_OUTPUT_DIM, THETA_OUTPUT_DIM))

        # self.base.head.fc = nn.Identity()
        #
        # for param in self.base.parameters():
        #     param.requires_grad = False
        #
        # self.head = nn.Sequential(nn.Linear(1536 + NO_CLUSTER, OUTPUT_DIM),
        #                           nn.GELU(),
        #                           nn.Linear(OUTPUT_DIM, OUTPUT_DIM))

    def forward(self, x, prob):
        x = self.base(x)
        model_input = torch.cat((x, prob), dim=1)
        y = self.head(model_input)
        return y


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device.type == 'cuda'
background = InaturalistEmbeddinga('evaluation', model_name='vitb32', model_family='clip')
pseudo_dset = InaturalistEmbeddinga('evaluation', model_name='vitg14')

phi_model_path = 'phi-dino-vitg14-Fully66D-finetuned-on_pseudoLabels-MIXED_TUNE=False-Animalia.pth'

theta_model_path = ROOT_PATH + '/experiments/persistents/' + 'CLIP-VITB32-224px-TwoLinear512-on66SC-OneHotConditionTraining-on-Animalia-650way-5shot_support-5shot_query-lr-0.0001.pth'

phi_model = nn.Linear(1536, NO_CLUSTER)
print(phi_model.load_state_dict(
    torch.load(ROOT_PATH + f'/experiments/persistents/{phi_model_path}', map_location=device)))

phi_model.to(device)
phi_model.eval()

# theta_model = TimmHead()
theta_model = nn.Sequential(nn.Linear(THETA_OUTPUT_DIM + NO_CLUSTER, THETA_OUTPUT_DIM),
                            nn.GELU(),
                            nn.Linear(THETA_OUTPUT_DIM, THETA_OUTPUT_DIM))

print(theta_model.load_state_dict(torch.load(theta_model_path, map_location=device)))
theta_model.float()
theta_model.to(device)
theta_model.eval()

meta_train_set = InaturalistEmbeddinga('background', model_name='vitg14')

super_class_prior = []
for super_cls in sorted(meta_train_set.df.super_class_name.unique()):
    base_classes = meta_train_set.df[meta_train_set.df['super_class_name'] == super_cls].class_id.unique()
    super_class_prior.append(len(base_classes))

super_class_prior = torch.tensor(super_class_prior)
super_class_prior = super_class_prior / torch.sum(super_class_prior)
super_class_prior = super_class_prior.to(device)

with torch.no_grad():
    class_embeddings = dict()
    pseudo_super_class = dict()
    for cls_id in tqdm(sorted(background.df['class_id'].unique())):
        cls_df = background.df[background.df['class_id'] == cls_id][:N_TRAIN]
        embeddings = []
        image_list = []
        phi_image_list = []
        for _, row in cls_df.iterrows():
            image, label = background[row['id']]
            image = image.to(device, dtype=torch.float32)

            phi_image, _ = pseudo_dset[row['id']]
            phi_image = phi_image.to(device, dtype=torch.float32)
            phi_image_list.append(phi_image)

            image_list.append(image)

        phi_input = torch.stack(phi_image_list)
        theta_input = torch.stack(image_list)

        super_class_logit = phi_model(phi_input)

        # logit = super_class_logit.log_softmax(dim=-1)
        # pseudo_label = torch.argmax(logit, dim=1).mode()[0].item()

        logit = super_class_logit.log_softmax(dim=-1).sum(dim=0)
        logit = logit - ((N_TRAIN - 1) * super_class_prior.log())

        pseudo_label = torch.argmax(logit).item()
        if pseudo_label in pseudo_super_class:
            pseudo_super_class[pseudo_label].append(cls_id)
        else:
            pseudo_super_class[pseudo_label] = [cls_id]

        # _, cluster_num_list = logit.topk(K, largest=True)
        #
        # for idx in cluster_num_list:
        #     if idx.item() in pseudo_super_class:
        #         pseudo_super_class[idx.item()].append(cls_id)
        #     else:
        #         pseudo_super_class[idx.item()] = [cls_id]

        sample_condition = nn.functional.one_hot(torch.tensor(pseudo_label), NO_CLUSTER).to(device)
        head_input = torch.cat((theta_input, sample_condition.repeat(N_TRAIN, 1)), dim=1)
        embeddings = theta_model(head_input)

        # embeddings = theta_model(theta_input, sample_condition.repeat(N_TRAIN, 1))

        class_embeddings[cls_id] = embeddings.mean(dim=0).detach().cpu()

class_prototypes = dict(sorted(class_embeddings.items()))
prototypes = torch.stack(list(class_prototypes.values()))

with open(ROOT_PATH + '/experiments/persistents/prototypes.pkl', 'wb') as handle:
    pickle.dump(prototypes.detach().cpu(), handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(ROOT_PATH + '/experiments/persistents/pseudo_super_class.pkl', 'wb') as handle:
    pickle.dump(pseudo_super_class, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(ROOT_PATH + '/experiments/persistents/prototypes.pkl', 'rb') as handle:
    prototypes = pickle.load(handle)

with open(ROOT_PATH + '/experiments/persistents/pseudo_super_class.pkl', 'rb') as handle:
    pseudo_super_class = pickle.load(handle)

prototypes = prototypes.to(device)

cluster_to_classes = []
for super_cls in range(len(pseudo_super_class.keys())):
    base_classes = pseudo_super_class[super_cls]
    base_classes.sort()
    cluster_to_classes.append(base_classes)

# inference all
acc = 0
tot = 0
with torch.no_grad():
    for cls_id in tqdm(sorted(background.df['class_id'].unique())):
        cls_df = background.df[background.df['class_id'] == cls_id][N_TRAIN:N_TRAIN + 5]
        for _, row in cls_df.iterrows():
            image, label = background[row['id']]
            label = torch.tensor(label)
            label = label.unsqueeze(0)
            label = label.to(device)

            phi_image, _ = pseudo_dset[row['id']]

            image = image.to(device, dtype=torch.float32)
            phi_image = phi_image.to(device, dtype=torch.float32)

            logits = phi_model(phi_image.unsqueeze(0))

            # sc_tensor = torch.argmax(logits, dim=1)
            # sample_condition = nn.functional.one_hot(sc_tensor, NO_CLUSTER).to(device)

            sample_condition, sample_idx = logits.topk(K, largest=True)
            prob = torch.zeros((1, NO_CLUSTER)).to(device)
            sample_condition = prob.scatter(1, sample_idx, sample_condition)
            sample_condition = torch.softmax(sample_condition, dim=1)
            head_input = torch.cat((image.unsqueeze(0), sample_condition), dim=1)

            # # sample_condition = nn.functional.one_hot(torch.tensor(pseudo_label), NO_CLUSTER).to(device)
            # # head_input = torch.cat((image.unsqueeze(0), sample_condition.unsqueeze(0)), dim=1)

            emb = theta_model(head_input)

            # emb = theta_model(image.unsqueeze(0), sample_condition)

            # emb = image.unsqueeze(0)

            _, cluster_num_list = logits.topk(K, largest=True)
            base_classes_label = []
            for idx in cluster_num_list[0]:
                base_classes_label.extend(cluster_to_classes[idx])

            base_classes_label = list(set(base_classes_label))

            cluster_prototypes = prototypes[base_classes_label]

            distances = (emb.repeat((cluster_prototypes.shape[0], 1)) - cluster_prototypes).pow(2).sum(dim=-1)
            distances = distances.view(-1, cluster_prototypes.shape[0])

            # y_pred = torch.argmax(-distances, dim=1)  # (q_test * k_test,)
            _, y_pred = torch.topk(-distances, min(TOP_ACC, len(cluster_prototypes)), dim=1)  # (q_test * k_test,)
            y_pred = y_pred[0]

            y_pred_list = list()
            for idx in y_pred:
                y_pred_list.append(base_classes_label[idx])

            if TOP_ACC != 1 and label.item() in y_pred_list:
                acc += 1
            elif TOP_ACC == 1 and label == base_classes_label[y_pred]:
                acc += 1

            tot += 1

    print(acc)
    print(tot)
    print(acc / tot)
    # writer.add_scalar('Average class Accuracy', running_acc / len(evaluation.df['class_id'].unique()), 1)
