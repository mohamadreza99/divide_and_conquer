from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
import os

seed = 100

# title = 'iNat Animalia t-SNE Embeddings'
title = 'iNat Animalia t-SNE Embeddings'

pic_path = title + '/' + str(seed) + '/'

if not os.path.exists(pic_path):
    os.makedirs(pic_path)

cmap = plt.get_cmap('jet')

N_select_sup = 74
N_final_select = 74
N_samples = 50
same_color_super = True
perplexity = 100

train_data = np.load('Animalia_DINOG_background_condition_embeddings.pkl', allow_pickle=True)
eval_data = np.load('Animalia_DINOG_evaluation_condition_embeddings.pkl', allow_pickle=True)

train_keys = list(train_data.keys())
eval_keys = list(eval_data.keys())

train_embs = np.array([train_data[key].numpy() for key in train_keys])
eval_embs = np.array([eval_data[key].numpy() for key in eval_keys])


def filter_keys(keys):
    sup_keys = []
    base_keys = []
    for key in keys:
        underscores = [pos for pos, char in enumerate(key) if char == '_']
        superclass = key[underscores[4] + 1: underscores[5]]
        baseclass = key[underscores[6] + 1:]

        sup_keys.append(superclass)
        base_keys.append(baseclass)

    return np.array(sup_keys), np.array(base_keys)


train_sup_keys, train_base_keys = filter_keys(train_keys)
eval_sup_keys, eval_base_keys = filter_keys(eval_keys)

sup_names = list(set(train_sup_keys))

selected_supers = np.random.choice(sup_names, size=N_select_sup, replace=False)

selected_train_inds = [np.argwhere(train_sup_keys == key).ravel() for key in selected_supers]
selected_eval_inds = [np.argwhere(eval_sup_keys == key).ravel() for key in selected_supers]

selected_train_inds_rvl = np.concatenate(selected_train_inds)
selected_eval_inds_rvl = np.concatenate(selected_eval_inds)

concat_embs = np.concatenate([train_embs[selected_train_inds_rvl], eval_embs[selected_eval_inds_rvl]])

concat_embs_rs = concat_embs.reshape((-1, concat_embs.shape[-1]))

tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity)
tsne_embs = tsne.fit_transform(concat_embs_rs)

tsne_embs_rs = tsne_embs.reshape((-1, N_samples, 2))

tsne_protos = tsne_embs_rs.mean(1)
tsne_dists = np.sqrt(np.square((tsne_embs_rs - tsne_protos[:, np.newaxis]).sum(-1)))

rdc_tsne = []

for k in range(len(tsne_embs_rs)):
    dists = tsne_dists[k]
    sorted_dists = np.argsort(dists)
    selected_dists = np.random.choice(sorted_dists[2:10], size=5, replace=False).ravel()
    rdc_tsne.append(tsne_embs_rs[k, selected_dists])

rdc_tsne = np.array(rdc_tsne)
tsne_protos = rdc_tsne.mean(1)

train_tsne_embs = []
eval_tsne_embs = []

indicator = 0
for inds in selected_train_inds:
    l_inds = len(inds)
    sup_embs = rdc_tsne[indicator: indicator + l_inds]
    train_tsne_embs.append(sup_embs)
    indicator += l_inds

for inds in selected_eval_inds:
    l_inds = len(inds)
    sup_embs = rdc_tsne[indicator: indicator + l_inds]
    eval_tsne_embs.append(sup_embs)
    indicator += l_inds

all_tsne_embs = [np.concatenate([train_tsne_embs[k], eval_tsne_embs[k]]) for k in range(len(train_tsne_embs))]

tsne_super_stds = []
train_tsne_embs_selected_base = []
eval_tsne_embs_selected_base = []
train_selected_base_inds = []
eval_selected_base_inds = []

for j in range(len(all_tsne_embs)):
    embs = all_tsne_embs[j]
    train_embs = train_tsne_embs[j]
    eval_embs = eval_tsne_embs[j]

    base_prototypes = embs.mean(1).mean(0)

    train_dists = np.sqrt(np.square((train_embs - base_prototypes[np.newaxis, np.newaxis]).sum(-1).sum(-1)))

    sorted_dists = np.argsort(train_dists)
    N1 = max(len(train_dists) // 3, 8)
    N2 = max(len(train_dists) // 4, 5)
    selected_dists = np.random.choice(sorted_dists[:N1], size=N2, replace=False).ravel()

    train_selected_base_inds.append(selected_dists)
    train_tsne_embs_selected_base.append(train_embs[selected_dists])

    eval_dists = np.sqrt(np.square((eval_embs - base_prototypes[np.newaxis, np.newaxis]).sum(-1).sum(-1)))
    sorted_dists = np.argsort(eval_dists)
    N1 = max(len(eval_dists) // 3, 8)
    N2 = max(len(eval_dists) // 4, 5)
    selected_dists = np.random.choice(sorted_dists[:N1], size=N2, replace=False).ravel()

    eval_selected_base_inds.append(selected_dists)
    eval_tsne_embs_selected_base.append(eval_embs[selected_dists])

    new_embs = np.concatenate((train_embs[selected_dists], eval_embs[selected_dists]))
    new_base_prototypes = new_embs.mean(1)
    dists = np.sqrt(np.square((new_embs - new_base_prototypes[:, np.newaxis]).sum(-1)))
    std = np.std(dists)
    tsne_super_stds.append(std)

skip_inds = [0]
selected_supers_ind = np.argsort(tsne_super_stds)[:N_final_select]
selected_supers_ind = np.delete(selected_supers_ind, [skip_inds])
# selected_train_embs = train_tsne_embs_selected_base[selected_supers_ind]
# selected_eval_embs = eval_tsne_embs_selected_base[selected_supers_ind]

if same_color_super:

    colors = np.arange(N_final_select)

    colors = cmap(colors / np.max(colors) * 5)

    plt.figure(figsize=(10, 10))
    for j, super_ind in enumerate(selected_supers_ind):
        train_embs = train_tsne_embs_selected_base[super_ind]
        for base_embs in train_embs:
            hull = ConvexHull(base_embs)
            for simplex in hull.simplices:
                plt.plot(base_embs[simplex, 0], base_embs[simplex, 1], 'c')

        conc_train_embs = np.concatenate(train_embs)
        plt.scatter(conc_train_embs[:, 0], conc_train_embs[:, 1], marker='o', s=10, c=colors[j])

        eval_embs = eval_tsne_embs_selected_base[super_ind]
        for base_embs in eval_embs:
            hull = ConvexHull(base_embs)
            for simplex in hull.simplices:
                plt.plot(base_embs[simplex, 0], base_embs[simplex, 1], 'r')

        conc_eval_embs = np.concatenate(eval_embs)
        plt.scatter(conc_eval_embs[:, 0], conc_eval_embs[:, 1], marker='*', s=10, c=colors[j])

if False:

    plt.figure(figsize=(10, 10))

    train_base_embs = np.concatenate([train_tsne_embs_selected_base[super_ind] for super_ind in selected_supers_ind])
    eval_base_embs = np.concatenate([eval_tsne_embs_selected_base[super_ind] for super_ind in selected_supers_ind])

    colors = [5 * [i] for i in range(len(train_base_embs))]
    colors = np.concatenate(colors)
    colors = cmap(colors / np.max(colors) * 5)
    train_base_embs = train_base_embs.reshape((-1, 2))

    plt.scatter(train_base_embs[:, 0], train_base_embs[:, 1], marker='o', s=10, c=colors)

    colors = [5 * [i] for i in range(len(eval_base_embs))]
    colors = np.concatenate(colors)
    colors = cmap(colors / np.max(colors) * 5)
    eval_base_embs = eval_base_embs.reshape((-1, 2))

    plt.scatter(eval_base_embs[:, 0], eval_base_embs[:, 1], marker='*', s=10, c=colors)

    for super_ind in selected_supers_ind:

        super_embs = np.concatenate(
            np.concatenate((train_tsne_embs_selected_base[super_ind], eval_tsne_embs_selected_base[super_ind])))
        hull = ConvexHull(super_embs)
        for simplex in hull.simplices:
            plt.plot(super_embs[simplex, 0], super_embs[simplex, 1], 'c')

    plt.savefig('embs.png', dpi=160)

if True:

    plt.figure(figsize=(10, 10))

    train_base_embs = np.concatenate([train_tsne_embs_selected_base[super_ind] for super_ind in selected_supers_ind])
    eval_base_embs = np.concatenate([eval_tsne_embs_selected_base[super_ind] for super_ind in selected_supers_ind])

    colors = [1 * [i] for i in range(len(train_base_embs))]
    colors = np.concatenate(colors)
    colors = cmap(colors / np.max(colors) * 1)
    train_base_embs = train_base_embs.mean(1)

    plt.scatter(train_base_embs[:, 0], train_base_embs[:, 1], marker='o', s=15, c=colors)

    colors = [1 * [i] for i in range(len(eval_base_embs))]
    colors = np.concatenate(colors)
    colors = cmap(colors / np.max(colors) * 1)
    eval_base_embs = eval_base_embs.mean(1)

    plt.scatter(eval_base_embs[:, 0], eval_base_embs[:, 1], marker='*', s=15, c=colors)

    for super_ind in selected_supers_ind:

        super_embs = np.concatenate(
            (train_tsne_embs_selected_base[super_ind].mean(1), eval_tsne_embs_selected_base[super_ind].mean(1)))
        hull = ConvexHull(super_embs)
        for simplex in hull.simplices:
            plt.plot(super_embs[simplex, 0], super_embs[simplex, 1], 'c', linewidth=.5)
    plt.tight_layout()
    plt.savefig('embs2.png', dpi=160)


def rotate(p, origin=(0, 0), degrees=45):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


scale = 1

if True:

    for rep in range(10):
        plt.figure(figsize=(9, 9))

        train_base_embs = np.concatenate(
            [train_tsne_embs_selected_base[super_ind] for super_ind in selected_supers_ind])
        eval_base_embs = np.concatenate([eval_tsne_embs_selected_base[super_ind] for super_ind in selected_supers_ind])

        train_base_embs = train_base_embs.mean(1)
        eval_base_embs = eval_base_embs.mean(1)

        train_base_embs = rotate(train_base_embs)
        eval_base_embs = rotate(eval_base_embs)

        train_base_embs[:, 0] = train_base_embs[:, 0] * scale
        eval_base_embs[:, 0] = eval_base_embs[:, 0] * scale

        # colors = [1 * [i] for i in range(len(train_base_embs))]
        # colors = np.concatenate(colors)
        # colors = cmap(colors / np.max(colors) * 1)

        plt.scatter(train_base_embs[:, 0], train_base_embs[:, 1], marker='o', s=15, label='source base-class')

        # colors = [1 * [i] for i in range(len(eval_base_embs))]
        # colors = np.concatenate(colors)
        # colors = cmap(colors / np.max(colors) * 1)

        plt.scatter(eval_base_embs[:, 0], eval_base_embs[:, 1], marker='*', s=15, label='target base-class')

        colors = np.random.permutation(N_final_select)
        colors = cmap(colors / np.max(colors) * 1)

        for j, super_ind in enumerate(selected_supers_ind):

            super_embs = np.concatenate(
                (train_tsne_embs_selected_base[super_ind].mean(1), eval_tsne_embs_selected_base[super_ind].mean(1)))
            super_embs = rotate(super_embs)
            super_embs[:, 0] = super_embs[:, 0] * scale

            hull = ConvexHull(super_embs)
            for simplex in hull.simplices:
                plt.plot(super_embs[simplex, 0], super_embs[simplex, 1], c=colors[j], linewidth=.8)
        # plt.tight_layout()
        plt.title(title)
        plt.legend()
        plt.savefig(pic_path + 'embs' + str(rep) + '.png', dpi=160)
