import sys

import torch
import torch.nn as nn
from tqdm import tqdm
import timm

sys.path.append("../..")
from data_loading.datasets import InaturalistEmbedding
from utilites.utils import pairwise_distances
from config import ROOT_PATH
import pickle

THETA_OUTPUT_DIM = 512
NO_CLUSTER = 74
N_TRAIN = 5
K = 10
TOP_ACC = 5

print('k:', K)
print('top acc:', TOP_ACC)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device.type == 'cuda'
background = InaturalistEmbedding('evaluation',model_family='clip' ,model_name='vitb32')
pseudo_dset = InaturalistEmbedding('evaluation', model_name='vitg14')

phi_model_path = 'phi-dino-vitg14-Fully66D-finetuned-on_pseudoLabels-MIXED_TUNE=False-Animalia.pth'

theta_model_path = ROOT_PATH + '/experiments/persistents/' + 'new-prototypical-baseline-CLIP-VITB32-224px-TwoLinear512-66SC-on-Animalia-650way-5shot_support-5shot_query-lr-0.0001.pth'

phi_model = nn.Linear(1536, NO_CLUSTER)
print(phi_model.load_state_dict(
    torch.load(ROOT_PATH + f'/experiments/persistents/{phi_model_path}', map_location=device)))

phi_model.to(device)
phi_model.eval()

model_head = nn.Sequential(nn.Linear(THETA_OUTPUT_DIM, THETA_OUTPUT_DIM),
                           nn.GELU(),
                           nn.Linear(THETA_OUTPUT_DIM, THETA_OUTPUT_DIM))

theta_model = model_head

# theta_model = timm.create_model('mobilenetv3_large_100.miil_in21k', pretrained=True)
# # model.classifier = nn.Identity()
# theta_model.classifier = model_head

# theta_model = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384',
#                                 pretrained=True)  # Huge = 2816 , Large = 1536
# theta_model.head.fc = nn.Identity()
# theta_model.head.fc = model_head

# theta_model = nn.Sequential(nn.Linear(1536, THETA_OUTPUT_DIM),
#                             nn.GELU(),
#                             nn.Linear(THETA_OUTPUT_DIM, THETA_OUTPUT_DIM))

print(theta_model.load_state_dict(torch.load(theta_model_path, map_location=device)))
theta_model.float()
theta_model.to(device)
theta_model.eval()

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

        super_class_logit = phi_model(phi_input)

        logit = super_class_logit.log_softmax(dim=-1).sum(dim=0)

        pseudo_label = torch.argmax(logit).item()
        if pseudo_label in pseudo_super_class:
            pseudo_super_class[pseudo_label].append(cls_id)
        else:
            pseudo_super_class[pseudo_label] = [cls_id]

        theta_input = torch.stack(image_list)
        embeddings = theta_model(theta_input)
        # embeddings = theta_input

        class_embeddings[cls_id] = embeddings.mean(dim=0).detach().cpu()

class_prototypes = dict(sorted(class_embeddings.items()))
prototypes = torch.stack(list(class_prototypes.values()))
prototypes = prototypes.to(device)

cluster_to_classes = []
for super_cls in range(len(pseudo_super_class.keys())):
    base_classes = pseudo_super_class[super_cls]
    base_classes.sort()
    cluster_to_classes.append(base_classes)

# supervised superclass (oracle phi)
# cluster_to_classes = []
# for super_cls in sorted(background.df.super_class_name.unique()):
#     base_classes = background.df[background.df['super_class_name'] == super_cls].class_id.unique()
#     base_classes.sort()
#     cluster_to_classes.append(base_classes)

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

            emb = theta_model(image.unsqueeze(0))
            # emb = image.unsqueeze(0)

            # cluster_num = torch.argmax(logits, dim=1).item()

            # supervised superclass (oracle phi)
            # for ii, cl in enumerate(cluster_to_classes):
            #     if label.item() in cl:
            #         cluster_num = ii
            #         break

            _, cluster_num_list = logits.topk(K, largest=True)
            base_classes_label = []
            for idx in cluster_num_list[0]:
                base_classes_label.extend(cluster_to_classes[idx])

            base_classes_label = list(set(base_classes_label))

            cluster_prototypes = prototypes[base_classes_label]

            distances = (emb.repeat((cluster_prototypes.shape[0], 1)) - cluster_prototypes).pow(2).sum(dim=-1)
            distances = distances.view(-1, cluster_prototypes.shape[0])

            # y_pred = torch.argmax(-distances, dim=1)  # (q_test * k_test,)
            _, y_pred = torch.topk(-distances, TOP_ACC, dim=1)  # (q_test * k_test,)
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
