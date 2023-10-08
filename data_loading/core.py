import torch
from torch.utils.data import Sampler
from typing import List, Iterable, Callable, Tuple
import numpy as np


class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,  # -> n shot task
                 k: int = None,  # -> k ways in support set
                 q: int = None,  # -> number of samples in class per query set, query set is k-way too.
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None,
                 hardMode: bool = False,
                 cluster_list=None,  # numpy array
                 ratio_of_hard_cluster: float = 0.8,
                 ):

        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        self.i_task = 0

        self.hardMode = hardMode
        if self.hardMode:
            self.cluster_list = cluster_list
            self.ratio_of_hard_cluster = ratio_of_hard_cluster

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.hardMode:
                    selected_cluster = np.random.choice(np.unique(self.cluster_list), size=1)
                    indx_of_classes_in_hard_cluster = np.argwhere(self.cluster_list == selected_cluster).flatten()
                    other_classes = self.dataset.df[
                        ~self.dataset.df['class_id'].isin(indx_of_classes_in_hard_cluster)]
                    other_classes = np.random.choice(other_classes['class_id'].unique(),
                                                     size=self.k - int(self.k * self.ratio_of_hard_cluster))
                    episode_classes = np.random.choice(indx_of_classes_in_hard_cluster,
                                                       size=int(self.k * self.ratio_of_hard_cluster))
                    episode_classes = np.concatenate((episode_classes, other_classes))

                else:
                    # Get random classes
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k,
                                                           replace=False)

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    # Select support examples
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support

                    for i, s in support.iterrows():
                        batch.append(s['id'])

                for k in episode_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])

            yield np.stack(batch)
