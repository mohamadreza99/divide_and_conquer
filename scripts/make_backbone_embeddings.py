import sys
import torch
from tqdm import tqdm
import pickle
import timm
import clip

sys.path.append("..")
from data_loading.datasets import InaturalistPlantae
from config import ROOT_PATH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device.type == 'cuda'

model, preprocess = clip.load("ViT-B/32", device=device)
model = model.encode_image

# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')  # testing on VIT Large
# model = timm.create_model('mobilenetv3_large_100.miil_in21k', pretrained=True)
# model.classifier = nn.Identity()

# model = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True)  # Huge = 2816 , Large = 1536
# model.head.fc = nn.Identity()

NAME_STRING = 'clip_vitb32_embeddings_dict.pkl'

# model.to(device)
# model.eval()

background = InaturalistPlantae('background')
evaluation = InaturalistPlantae('evaluation')

print(f"len of all background images are {len(background)}")
print(f"len of all evaluation images are {len(evaluation)}")


def get_embeddings(dataset):
    with torch.no_grad():
        class_embeddings = {}
        for cls_id in tqdm(dataset.df['class_id'].unique()):
            cls_df = dataset.df[dataset.df['class_id'] == cls_id]
            images = []
            for _, row in cls_df.iterrows():
                image, label = dataset[row['id']]
                image = image.to(device)
                images.append(image)

            model_input = torch.stack(images)
            emb = model(model_input)

            class_embeddings[cls_id] = emb.detach().cpu().numpy()
            torch.cuda.empty_cache()

    embeddings_dict = dict(sorted(class_embeddings.items()))
    return embeddings_dict


embeddings_dict = get_embeddings(background)
# prototypes = torch.stack(list(class_prototypes.values()))

with open(
        ROOT_PATH + '/experiments/persistents/Plantae_background_' + NAME_STRING,
        'wb') as handle:
    pickle.dump(embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('saved')

embeddings_dict = get_embeddings(evaluation)
# prototypes = torch.stack(list(class_prototypes.values()))

with open(
        ROOT_PATH + '/experiments/persistents/Plantae_evaluation_' + NAME_STRING,
        'wb') as handle:
    pickle.dump(embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('saved')


# read dictionary from pickl file

with open(ROOT_PATH + '/experiments/persistents/Plantae_background_' + NAME_STRING, 'rb') as handle:
    background_embeddings_dict = pickle.load(handle)

with open(ROOT_PATH + '/experiments/persistents/Plantae_evaluation_' + NAME_STRING, 'rb') as handle:
    evaluation_embeddings_dict = pickle.load(handle)

# sort all keys then stack all dictionary values as one numpy array
background_embeddings = dict()
for key in sorted(background_embeddings_dict.keys()):
    name = background.id_to_class_name[key]
    background_embeddings[name] = background_embeddings_dict[key]

evaluation_embeddings = dict()
for key in sorted(evaluation_embeddings_dict.keys()):
    name = evaluation.id_to_class_name[key]
    evaluation_embeddings[name] = evaluation_embeddings_dict[key]

# save the dictionary in pick file in same directory as the ct file
with open(ROOT_PATH + '/experiments/persistents/Plantae_background_keyname_' + NAME_STRING, 'wb') as handle:
    pickle.dump(background_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('saved')

with open(ROOT_PATH + '/experiments/persistents/Plantae_evaluation_keyname_' + NAME_STRING, 'wb') as handle:
    pickle.dump(evaluation_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('saved')

