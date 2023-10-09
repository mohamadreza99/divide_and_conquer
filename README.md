<br><br><br>

# Divide and Conquer: Two-Level Problem Remodeling for Large-Scale Few-Shot Learning (Submitted to R0-FoMo Workshop - Neurips 2023)

In this repository we provide PyTorch implementations for Divide and Conquer: Two-Level Problem Remodeling for Large-Scale Few-Shot Learning. The directory outline is as follows:

```bash
root
├── README.md
├── config.py
├── data_loading
│   ├── __init__.py
│   ├── core.py
│   └── datasets.py
├── experiments
│   ├── __init__.py
│   ├── coarse_grained
│   │   ├── __init__.py
│   │   ├── semi_supervised_fine_tuning.py
│   │   └── train_phi.py
│   └── fine_grained
│       ├── __init__.py
│       ├── baseline_evaluation.py
│       ├── evaluate_conditioned_theta.py
│       ├── topk_superclass_hierarchical_evaluation.py
│       ├── train_conditioned_theta_prototypical.py
│       └── train_theta_prototypical.py
├── scripts
│   ├── __init__.py
│   ├── make_backbone_embeddings.py
│   ├── prepare_dataset_animalia.py
│   └── prepare_dataset_plantae.py
├── tsne_visualization
│   ├── __init__.py
│   └── visualize_tsne.py
└── utilites
    ├── MeanStdDataset.py
    ├── __init__.py
    └── utils.py

```
In the following sections we will first provide details about how to setup the dataset. Then the instructions for installing package dependencies, training and testing is provided.

# Configuring the Dataset