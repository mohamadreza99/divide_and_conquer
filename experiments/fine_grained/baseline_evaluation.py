import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
# import clip
import timm

sys.path.append("../..")
from data_loading.datasets import InaturalistEmbedding
from config import ROOT_PATH

param_string = "Evaluation-prototypical-on-Omniglot-metatest-5shot-support-5shot-query"

TOP_ACC = 5
N_TRAIN = 5
OUTPUT_DIM = 512
NO_CLUSTER = 74

print('top acc:', TOP_ACC)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device.type == 'cuda'

model_head = nn.Sequential(nn.Linear(OUTPUT_DIM, OUTPUT_DIM),
                           nn.GELU(),
                           nn.Linear(OUTPUT_DIM, OUTPUT_DIM))

model = model_head
#
# model = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True)  # Huge = 2816 , Large = 1536
# model.head.fc = nn.Identity()
# model.head.fc = model_head

# model = timm.create_model('mobilenetv3_large_100.miil_in21k', pretrained=True)
# model.classifier = nn.Identity()
# model.classifier = model_head

model_head_path = ROOT_PATH + '/experiments/persistents/' + 'new-prototypical-baseline-CLIP-VITB32-224px-TwoLinear512-66SC-on-Animalia-650way-5shot_support-5shot_query-lr-0.0001.pth'
print(model.load_state_dict(torch.load(model_head_path, map_location=device)))

model.to(device)
model.eval()

evaluation = InaturalistEmbedding('evaluation', model_family='clip', model_name='vitb32')

print(f"len of all evaluation images are {len(evaluation)}")

with torch.no_grad():
    class_embeddings = {}
    for cls_id in tqdm(evaluation.df['class_id'].unique()):
        cls_df = evaluation.df[evaluation.df['class_id'] == cls_id][:N_TRAIN]
        embeddings = []
        for _, row in cls_df.iterrows():
            image, label = evaluation[row['id']]
            image = image.to(device, dtype=torch.float32)
            # emb = model(image.unsqueeze(0))
            emb = image.unsqueeze(0)

            embeddings.append(emb.cpu())
        embeddings = torch.concat(embeddings).to('cpu')
        class_embeddings[cls_id] = embeddings.mean(dim=0).cpu()
        # del embeddings
        torch.cuda.empty_cache()

class_prototypes = dict(sorted(class_embeddings.items()))
prototypes = torch.stack(list(class_prototypes.values()))

prototypes = prototypes.to(device)

# prototypes /= prototypes.norm(dim=1, keepdim=True)

# inference
acc = 0
total = 0
# model.eval()

with torch.no_grad():
    for cls_id in tqdm(sorted(evaluation.df['class_id'].unique())):
        class_loss = 0.0
        class_acc = 0.0
        cls_df = evaluation.df[evaluation.df['class_id'] == cls_id][N_TRAIN:N_TRAIN + 5]
        for _, row in cls_df.iterrows():
            image, label = evaluation[row['id']]
            label = torch.tensor(label)
            label = label.unsqueeze(0)
            label = label.to(device)

            image = image.to(device, dtype=torch.float32)
            # queries = model(image.unsqueeze(0))
            queries = image.unsqueeze(0)

            # queries /= queries.norm(dim=1, keepdim=True)

            # sim = (queries * prototypes).sum(dim=-1)

            sim = (queries - prototypes).pow(2).sum(dim=-1) * -1

            # y_pred = torch.argmax(sim, dim=0)
            _, y_pred = torch.topk(sim, TOP_ACC, dim=0)

            if label in y_pred:
                acc += 1

            total += 1

            torch.cuda.empty_cache()

    print(acc)
    print(total)
    print(acc / total)
