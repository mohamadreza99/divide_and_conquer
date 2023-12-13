# Divide and Conquer: Two-Level Problem Remodeling for Large-Scale Few-Shot Learning (Accepted at R0-FoMo Workshop - Neurips 2023)

In this repository we provide PyTorch implementations for Divide and Conquer: Two-Level Problem Remodeling for Large-Scale Few-Shot Learning. The directory outline is as follows:

```bash
root
├── README.md
├── config.py
├── data_loading
│   ├── __init__.py
│   ├── core.py
│   └── datasets.py
├── experiments
│   ├── __init__.py
│   ├── coarse_grained
│   │   ├── __init__.py
│   │   ├── semi_supervised_fine_tuning.py
│   │   └── train_phi.py
│   └── fine_grained
│       ├── __init__.py
│       ├── baseline_evaluation.py
│       ├── evaluate_conditioned_theta.py
│       ├── topk_superclass_hierarchical_evaluation.py
│       ├── train_conditioned_theta_prototypical.py
│       └── train_theta_prototypical.py
├── scripts
│   ├── __init__.py
│   ├── make_backbone_embeddings.py
│   ├── prepare_dataset_animalia.py
│   └── prepare_dataset_plantae.py
├── tsne_visualization
│   ├── __init__.py
│   └── visualize_tsne.py
└── utilites
    ├── MeanStdDataset.py
    ├── __init__.py
    └── utils.py
```

In the following sections we will first provide details about how to setup the dataset. Then the instructions for training and evaluating are provided.

## Dataset

This project utilizes the iNat2021-mini dataset, a subset of the iNaturalist 2021 dataset focused on species classification.

To use this dataset:

1. Download the iNat2021-mini images:
   - Images are available at https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz

2. Specify the path to your downloaded dataset in `config.py` by updating `DATA_PATH`.

3. Run data preprocessing:
   ```
   python scripts/prepare_dataset_animalia.py
   python scripts/prepare_dataset_plantae.py
   ```
   This will preprocess the data and create the final datasets for model training. 

4. (Optional) Extract backbone embeddings:
   ```
   python scripts/make_backbone_embeddings.py
   ```
   This will create backbone embeddings for any backbone models you wish to freeze during training. Doing this ahead of time saves compute during training.

## Training 

The main training scripts are located in the `experiments/` directory. 

To train a model:

1. Train the coarse-grained model:
   ```
   python experiments/coarse_grained/train_phi.py
   ```
   This trains the model to classify high-level categories. 
   
   - Specify the dataset in the script. The default is Animalia.
   - The default backbone is DINO ViT-G14, but you can change this by modifying the code.
   
2. Fine-tune on the meta-test set in a semi-supervised manner:
   ```
   python experiments/coarse_grained/semi_supervised_fine_tuning.py
   ```
   This fine-tunes the coarse model on new classes.
   
3. Train a prototypical fine-grained model:
   ```
   python experiments/fine_grained/train_theta_prototypical.py  
   ```
   This trains a baseline prototypical network.
   
4. Train a conditioned prototypical model:
   ```
   python experiments/fine_grained/train_conditioned_theta_prototypical.py
   ```
   This trains the proposed conditioned prototypical network.
   
Key points:

- Update the dataset and model directly in the code. Comments indicate available options.
- Check the arguments at the top of each script to modify training parameters. 
- You have to change the model name in `param_string` variable in training files to assign a proper name to your models.

## Evaluation

The evaluation scripts are located in `experiments/fine_grained`.

To evaluate models:

1. Evaluate a flat baseline model:
   ```
   python experiments/fine_grained/baseline_evaluation.py
   ```
   Before running, set the following in the script:
   - Dataset  
   - Model and path to load weights
   - Other global parameters in the files
   
2. Evaluate hierarchical inference:
   ```
   python experiments/fine_grained/topk_superclass_hierarchical_evaluation.py
   ```
   Important parameters:
   - Dataset, model, path to weights
   - `K` - number of superclasses to suggest per query  
   - Other global parameters in the files
   
3. Evaluate a conditioned model:
   ```
   python experiments/fine_grained/evaluate_conditioned_theta.py   
   ```
   Similar setup to hierarchical evaluation, just tests the conditioned model.

Key aspects of evaluation:

- Set the correct dataset, model, and path to weights in each script
- Adjust key parameters like K at the top of the file  
- Hierarchical evaluation suggests top K superclasses to enhance inference
- Conditioned evaluation directly tests the conditioned model
