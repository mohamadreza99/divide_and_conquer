'''
create shared super superclasses from level 5 between meta train and meta test
from a level5 super class, 1/2 of its base classes are moved to meta test
'''
import shutil
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.append('../')
from config import DATA_PATH

basePath = Path(DATA_PATH + '/dsets/inaturalist2021/train') # train folder for inaturalist2021mini

i = 0
superclass_dict = dict()
for cls in tqdm(sorted(basePath.iterdir())):
    l = cls.name.split('_')
    if l[1] == 'Plantae':
        super_class_name = l[5]
        if super_class_name in superclass_dict:
            i += 1
            superclass_dict[super_class_name].append(cls)
        else:
            superclass_dict[super_class_name] = []
            superclass_dict[super_class_name].append(cls)
            i += 1

for k, v in superclass_dict.copy().items():
    if 200 > len(v) > 15:
        continue
    del superclass_dict[k]

superclass_dict = dict(sorted(superclass_dict.items(), key=lambda x: len(x[1]), reverse=False))

train_list = list()
train_counter = 0

eval_list = list()
eval_counter = 0

for k,v in superclass_dict.items():
    level_6_class = list(map(lambda x : x.name.split('_')[6], v
         ))
    level_6_dict = Counter(level_6_class)
    i = 0
    single_flag = True
    while i < len(v):
        lvl_6_name = v[i].name.split('_')[6]
        lvl_6_count = level_6_dict[lvl_6_name]
        if lvl_6_count > 1:
            for j in range(lvl_6_count//2):
                eval_list.append((v[i]))
                eval_counter += 1
                i += 1
            for j in range(lvl_6_count//2, lvl_6_count):
                train_list.append(v[i])
                train_counter += 1
                i += 1
        elif single_flag:
            train_list.append(v[i])
            train_counter += 1
            i += 1
            single_flag = not single_flag
        else:
            eval_list.append((v[i]))
            eval_counter += 1
            i += 1
            single_flag = not single_flag

print(train_list)
print(len(train_list))
print(train_counter)

print(eval_list)
print(len(eval_list))
print(eval_counter)

assert len(train_list) == 1450
assert len(eval_list) == 1255

# now copy the selected classes to corresponding path
for cls in tqdm(train_list):
    files = list(cls.glob('*'))
    for indx in range(len(files)):
        f = files[indx]
        pastedir = Path(DATA_PATH + '/meta_dsets/plantae/images_background') / f.parent.name
        pastedir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(f), str(pastedir))

for cls in tqdm(eval_list):
    files = list(cls.glob('*'))
    for indx in range(len(files)):
        f = files[indx]
        pastedir = Path(DATA_PATH + '/meta_dsets/plantae/images_evaluation') / f.parent.name
        pastedir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(f), str(pastedir))

assert len(list(Path(DATA_PATH + '/meta_dsets/plantae/images_background').glob('*'))) == 1450
assert len(list(Path(DATA_PATH + '/meta_dsets/plantae/images_evaluation').glob('*'))) == 1255

path_dirs = [Path(DATA_PATH + '/meta_dsets/plantae/images_background'),
     Path(DATA_PATH + '/meta_dsets/plantae/images_evaluation')]
for p in path_dirs:
    for cls in p.iterdir():
        assert len(list(cls.glob('*'))) == 50


print('done')
