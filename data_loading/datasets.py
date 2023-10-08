import typing
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle

from config import DATA_PATH, ROOT_PATH


class InaturalistPlantae(Dataset):
    def __init__(self, subset, coarse_label=False):
        """Dataset class representing Inaturalist2021 dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise (ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        self.path_dirs = [Path(DATA_PATH + '/meta_dsets/plantae/images_background'),
                          Path(DATA_PATH + '/meta_dsets/plantae/images_evaluation')]
        self.coarse_label = coarse_label
        if self.subset == 'background':
            self.data_root = self.path_dirs[0]
        else:
            self.data_root = self.path_dirs[1]

        self.df = pd.DataFrame(self.index_subset(self.subset, self.data_root))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.id_to_class_name = {i: self.unique_characters[i] for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.502, 0.510, 0.423],  # Inat2021mini mean and std
                                 std=[0.241, 0.234, 0.261])
        ])

    def __getitem__(self, item):
        file_path = self.df.iloc[item, 2]
        instance = Image.open(file_path)
        instance = self.transform(instance)
        if self.coarse_label:
            label = self.df.iloc[item, 4]
        else:
            label = self.df.iloc[item, 6]
        return instance, label

    def superclass_index(self, label):
        """gives to coarse label based on fine label of a class"""
        return self.df[self.df['class_id'] == label].iloc[0]['super_class_id']

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    def num_super_classes(self):
        return len(self.df['super_class_name'].unique())

    @staticmethod
    def index_subset(subset, data_root):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Inaturalist2021 dataset dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        # change path to /meta_dsets/inat2021mini/images_{}/ for using inat on chordata 2416 fine classes
        for root, folders, files in os.walk(DATA_PATH + '/meta_dsets/plantae/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.jpg')])

        i = 0
        superclass_dict = dict()
        for cls in tqdm(sorted(data_root.iterdir())):
            l = cls.name.split('_')
            super_class_name = l[5]
            if super_class_name in superclass_dict:
                i += 1
                files = cls.glob('*')
                for image in files:
                    superclass_dict[super_class_name].append(image)
            else:
                superclass_dict[super_class_name] = []
                files = cls.glob('*')
                for image in files:
                    superclass_dict[super_class_name].append(image)
                i += 1

        print(i)

        superclasses = sorted(list(superclass_dict.keys()))
        all_files = []
        for super_cls in list(superclass_dict.values()):
            all_files.extend(super_cls)
        files = sorted(all_files)
        progress_bar = tqdm(total=subset_len)
        for f in files:
            progress_bar.update(1)
            class_name = f.parent.name
            super_class_name = f.parent.name.split('_')[5]
            images.append({
                'subset': subset,
                'class_name': class_name,
                'filepath': f,
                'super_class_name': super_class_name,
                'super_class_id': superclasses.index(super_class_name)
            })
        progress_bar.close()
        return images


class InaturalistPlantaeEmbedding(Dataset):
    def __init__(self, subset, model_family='dino', model_name='vitg14', coarse_label: typing.Optional[bool] = False):
        """Dataset class representing Inaturalist2021 dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise (ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        self.coarse_label = coarse_label
        self.data_root = Path(
            DATA_PATH + '/meta_dsets/inatemb/' + f'Plantae_{self.subset}_keyname_{model_family}_{model_name}_embeddings_dict.pkl')

        self.df = pd.DataFrame(self.index_subset(self.subset, self.data_root))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

    def __getitem__(self, item):
        # instance = self.datasetid_to_emb[item]
        instance = self.df.iloc[item, 2]
        instance = torch.tensor(instance, dtype=torch.float32)
        if self.coarse_label:
            label = self.df.iloc[item, 4]
        else:
            label = self.df.iloc[item, 6]
        return instance, label

    def superclass_index(self, label):
        """gives to coarse label based on fine label of a class"""
        return self.df[self.df['class_id'] == label].iloc[0]['super_class_id']

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    def num_super_classes(self):
        return len(self.df['super_class_name'].unique())

    @staticmethod
    def index_subset(subset, data_root):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Inaturalist2021 dataset dataset
        """
        with open(data_root, 'rb') as handle:
            emb_dict = pickle.load(handle)
        emb_dict = dict(sorted(emb_dict.items()))

        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        # change path to /meta_dsets/inat2021mini/images_{}/ for using inat on chordata 2416 fine classes
        superclasses = set()
        for k, v in emb_dict.items():
            superclasses.add(k.split('_')[5])
            subset_len += len(v)

        superclasses = sorted(list(superclasses))

        i = 0
        progress_bar = tqdm(total=subset_len)
        superclass_dict = dict()
        for cls, v in tqdm(emb_dict.items()):
            l = cls.split('_')
            super_class_name = l[5]
            files = list(v)
            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': cls,
                    'emb': f,
                    'super_class_name': super_class_name,
                    'super_class_id': superclasses.index(super_class_name)
                })
            i += 1
        progress_bar.close()
        print(f'no super classes = {len(superclasses)}')
        print(f'len classes = {i}')
        return images


class Inaturalist2021mini(Dataset):
    def __init__(self, subset, coarse_label=False):
        """Dataset class representing Inaturalist2021 dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise (ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        self.path_dirs = [Path(DATA_PATH + '/meta_dsets/animalia/images_background'),
                          Path(DATA_PATH + '/meta_dsets/animalia/images_evaluation')]
        self.coarse_label = coarse_label
        if self.subset == 'background':
            self.data_root = self.path_dirs[0]
        else:
            self.data_root = self.path_dirs[1]

        self.df = pd.DataFrame(self.index_subset(self.subset, self.data_root))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.id_to_class_name = {i: self.unique_characters[i] for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.502, 0.510, 0.423],  # Inat2021mini mean and std
                                 std=[0.241, 0.234, 0.261])
        ])

    def __getitem__(self, item):
        file_path = self.df.iloc[item, 2]
        instance = Image.open(file_path)
        instance = self.transform(instance)
        if self.coarse_label:
            label = self.df.iloc[item, 4]
        else:
            label = self.df.iloc[item, 6]
        return instance, label

    def superclass_index(self, label):
        """gives to coarse label based on fine label of a class"""
        return self.df[self.df['class_id'] == label].iloc[0]['super_class_id']

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    def num_super_classes(self):
        return len(self.df['super_class_name'].unique())

    @staticmethod
    def index_subset(subset, data_root):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Inaturalist2021 dataset dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        # change path to /meta_dsets/inat2021mini/images_{}/ for using inat on chordata 2416 fine classes
        for root, folders, files in os.walk(DATA_PATH + '/meta_dsets/animalia/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.jpg')])

        i = 0
        superclass_dict = dict()
        for cls in tqdm(sorted(data_root.iterdir())):
            l = cls.name.split('_')
            super_class_name = l[5]
            if super_class_name in superclass_dict:
                i += 1
                files = cls.glob('*')
                for image in files:
                    superclass_dict[super_class_name].append(image)
            else:
                superclass_dict[super_class_name] = []
                files = cls.glob('*')
                for image in files:
                    superclass_dict[super_class_name].append(image)
                i += 1

        print(i)

        superclasses = sorted(list(superclass_dict.keys()))
        all_files = []
        for super_cls in list(superclass_dict.values()):
            all_files.extend(super_cls)
        files = sorted(all_files)
        progress_bar = tqdm(total=subset_len)
        for f in files:
            progress_bar.update(1)
            class_name = f.parent.name
            super_class_name = f.parent.name.split('_')[5]
            images.append({
                'subset': subset,
                'class_name': class_name,
                'filepath': f,
                'super_class_name': super_class_name,
                'super_class_id': superclasses.index(super_class_name)
            })
        progress_bar.close()
        return images


class InaturalistEmbedding(Dataset):
    def __init__(self, subset, model_family='dino', model_name='vitg14', coarse_label: typing.Optional[bool] = False):
        """Dataset class representing Inaturalist2021 dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise (ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        self.coarse_label = coarse_label
        self.data_root = Path(
            DATA_PATH + '/meta_dsets/inatemb/' + f'{self.subset}_keyname_{model_family}_{model_name}_embeddings_dict.pkl')

        self.df = pd.DataFrame(self.index_subset(self.subset, self.data_root))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

    def __getitem__(self, item):
        # instance = self.datasetid_to_emb[item]
        instance = self.df.iloc[item, 2]
        instance = torch.tensor(instance, dtype=torch.float32)
        if self.coarse_label:
            label = self.df.iloc[item, 4]
        else:
            label = self.df.iloc[item, 6]
        return instance, label

    def superclass_index(self, label):
        """gives to coarse label based on fine label of a class"""
        return self.df[self.df['class_id'] == label].iloc[0]['super_class_id']

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    def num_super_classes(self):
        return len(self.df['super_class_name'].unique())

    @staticmethod
    def index_subset(subset, data_root):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Inaturalist2021 dataset dataset
        """
        with open(data_root, 'rb') as handle:
            emb_dict = pickle.load(handle)
        emb_dict = dict(sorted(emb_dict.items()))

        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        # change path to /meta_dsets/inat2021mini/images_{}/ for using inat on chordata 2416 fine classes
        superclasses = set()
        for k, v in emb_dict.items():
            superclasses.add(k.split('_')[5])
            subset_len += len(v)

        superclasses = sorted(list(superclasses))

        i = 0
        progress_bar = tqdm(total=subset_len)
        superclass_dict = dict()
        for cls, v in tqdm(emb_dict.items()):
            l = cls.split('_')
            super_class_name = l[5]
            files = list(v)
            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': cls,
                    'emb': f,
                    'super_class_name': super_class_name,
                    'super_class_id': superclasses.index(super_class_name)
                })
            i += 1
        progress_bar.close()
        print(f'no super classes = {len(superclasses)}')
        print(f'len classes = {i}')
        return images



