# Federated Learning for Skin Lesion Analysis

A multi-dataset federated learning study evaluating **FedAvg, Weighted FedAvg, FedProx, and Ditto** for medical image classification and segmentation under **non-IID client distributions**, with additional analysis of **heterogeneity severity, explainability, communication cost, and machine unlearning**.

---

## Overview

Federated learning (FL) enables multiple clients to collaboratively train machine learning models without directly sharing their private training data. In medical imaging, this is particularly relevant because patient data is often distributed across hospitals or institutions and cannot easily be centralized.

This project investigates how different federated aggregation strategies behave when client data distributions are heterogeneous.

Rather than evaluating federated learning on a single dataset or a single form of non-IIDness, the study evaluates:

- **Two skin-lesion classification datasets**
  - HAM10000
  - ISIC-2019

- **One skin-lesion segmentation dataset**
  - ISIC-2018 Task 1

- **Four federated learning strategies**
  - FedAvg
  - Weighted FedAvg
  - FedProx
  - Ditto

- **Different forms and severities of non-IID data**
  - Class-distribution heterogeneity for classification
  - Lesion-area heterogeneity for segmentation
  - Mild, moderate, and extreme non-IID ablations

- **Additional analyses**
  - Centralized vs federated performance
  - Local vs federated performance
  - Per-class/client federation gains
  - Grad-CAM explainability
  - CAM evolution across communication rounds
  - Communication cost
  - Machine unlearning

The main objective is not to identify a universally best federated learning algorithm, but to study **how the effectiveness of an aggregation strategy changes with the task and the type and severity of client heterogeneity**.

---

# Research Question

> **How do different federated learning aggregation strategies behave across medical imaging tasks and under different types and severities of non-IID client heterogeneity?**

The study specifically investigates whether an aggregation strategy that performs well for classification under class imbalance also performs well for segmentation when heterogeneity is defined using a different data characteristic.

---

# Experimental Framework

| Dataset          | Task           | Model                     | Clients | Non-IID Dimension  | FL Methods                              |
| ---------------- | -------------- | ------------------------- | ------: | ------------------ | --------------------------------------- |
| HAM10000         | Classification | ResNet-18                 |       3 | Class distribution | FedAvg, Weighted FedAvg, FedProx, Ditto |
| ISIC-2019        | Classification | ResNet-18                 |       3 | Class distribution | FedAvg, Weighted FedAvg, FedProx, Ditto |
| ISIC-2018 Task 1 | Segmentation   | U-Net + ResNet-18 encoder |       3 | Lesion area        | FedAvg, Weighted FedAvg, FedProx, Ditto |

The main experiments use **20 federated communication rounds** with **3 local epochs per client per round**.

---

# Datasets

## 1. HAM10000

HAM10000 is used for multi-class skin-lesion classification.

### Dataset characteristics

- Approximately 10,000 dermoscopic images
- 7 lesion classes:
  - `akiec`
  - `bcc`
  - `bkl`
  - `df`
  - `mel`
  - `nv`
  - `vasc`

### Data partitioning

Three clients are constructed using a disjoint-by-class-majority non-IID partition.

| Client   | Majority Classes | Samples |
| -------- | ---------------- | ------: |
| Client 0 | akiec + bcc      |   1,352 |
| Client 1 | bkl + df         |   1,482 |
| Client 2 | mel + nv         |   5,178 |

The main experiment uses:

- `majority_frac = 0.8`
- 20 federated rounds
- 3 local epochs

Gaussian client-specific image noise is also introduced:

- Client 0: σ = 0
- Client 1: σ = 20
- Client 2: σ = 10

### Model

**ResNet-18**, pretrained on ImageNet.

The final classification layer is replaced with:

```text
Linear(512 → 7)
```

Training uses:

- AdamW
- Learning rate: `1e-4`
- Weight decay: `1e-5`
- Class-weighted CrossEntropyLoss
- Gradient clipping: `max_norm = 1.0`
- WeightedRandomSampler for client-level class balancing

### HAM10000 results

| Method          |  Accuracy |  Macro F1 |    G-Mean | Balanced Accuracy |       IoU |
| --------------- | --------: | --------: | --------: | ----------------: | --------: |
| Best Local      |     0.746 |     0.540 |     0.440 |             0.595 |         — |
| FedAvg          | **0.804** | **0.653** |     0.615 |             0.626 | **0.870** |
| Weighted FedAvg |     0.747 |     0.481 |     0.354 |             0.450 |     0.457 |
| FedProx         |     0.798 |     0.628 |     0.570 |             0.610 |     0.514 |
| Ditto           |     0.794 |     0.643 | **0.618** |         **0.638** |     0.562 |
| Centralized     | **0.818** | **0.719** | **0.708** |         **0.716** |         — |

### Important finding

Federation provides substantial benefits for some rare classes, particularly `df`, while specialist clients can experience performance losses on their majority classes.

For FedAvg, the reported per-class gains relative to the best local model include:

- `df`: **+0.48**
- `bkl`: **+0.28**
- `nv`: **+0.06**
- `mel`: **+0.02**
- `akiec`: **−0.06**
- `vasc`: **−0.21**
- `bcc`: **−0.35**

This demonstrates the trade-off between global collaboration and client specialization.

---

# 2. ISIC-2019

ISIC-2019 is used for a larger multi-class skin-lesion classification experiment.

### Dataset characteristics

- Approximately 23,000 images
- 8 classes:
  - `AK`
  - `BCC`
  - `BKL`
  - `DF`
  - `MEL`
  - `NV`
  - `SCC`
  - `VASC`

### Client partition

| Client   | Majority Classes | Samples |
| -------- | ---------------- | ------: |
| Client 0 | AK + BCC + BKL   |   5,304 |
| Client 1 | DF + MEL + NV    |  11,772 |
| Client 2 | SCC + VASC       |   1,529 |

The main experiment uses:

- `majority_frac = 0.9`
- 20 rounds
- 3 local epochs
- Batch size: 16

The test set is streamed during evaluation to reduce memory usage.

### Model

The same **ImageNet-pretrained ResNet-18** architecture is used.

The final layer is configured for 8-class classification.

### ISIC-2019 results

| Method          |  Accuracy |  Macro F1 |    G-Mean | Balanced Accuracy |       IoU |
| --------------- | --------: | --------: | --------: | ----------------: | --------: |
| Best Local      |     0.581 |     0.380 |     0.050 |             0.469 |         — |
| FedAvg          |     0.720 |     0.549 |     0.485 |             0.516 |     0.422 |
| Weighted FedAvg |     0.685 |     0.426 |     0.291 |             0.396 |     0.317 |
| FedProx         |     0.708 |     0.550 | **0.504** |             0.534 | **0.529** |
| Ditto           | **0.725** | **0.561** |     0.501 |         **0.538** |     0.497 |
| Centralized     | **0.736** | **0.654** | **0.636** |         **0.647** |         — |

The data-poor Client 2 has a very low local G-Mean of approximately **0.05**. Federation substantially improves its general classification capability, with Ditto reaching approximately **0.50 G-Mean**.

---

# Non-IID Severity Ablation

To study how the algorithms respond as heterogeneity increases, the classification experiments use three levels of majority-class concentration.

| Severity | Majority Fraction |
| -------- | ----------------: |
| Mild     |               0.5 |
| Moderate |               0.7 |
| Extreme  |         0.8 / 0.9 |

## HAM10000 — Macro F1

| Method          |  Mild |  Moderate |   Extreme |
| --------------- | ----: | --------: | --------: |
| FedAvg          | 0.620 |     0.600 | **0.653** |
| Weighted FedAvg | 0.574 |     0.503 |     0.481 |
| FedProx         | 0.610 |     0.596 |     0.628 |
| Ditto           | 0.609 | **0.641** |     0.643 |

## ISIC-2019 — Macro F1

| Method          |  Mild | Moderate |   Extreme | Degradation |
| --------------- | ----: | -------: | --------: | ----------: |
| FedAvg          | 0.604 |    0.588 |     0.549 |       −9.1% |
| Weighted FedAvg | 0.597 |    0.551 |     0.426 |  **−28.6%** |
| FedProx         | 0.607 |    0.584 |     0.550 |       −9.4% |
| Ditto           | 0.599 |    0.579 | **0.561** |   **−6.4%** |

The corresponding G-Mean degradation for ISIC-2019 is:

- FedAvg: −13.5%
- Weighted FedAvg: **−47.5%**
- FedProx: −7.9%
- Ditto: −10.7%

### Finding

Weighted FedAvg becomes particularly vulnerable when client dataset size is correlated with class composition.

Ditto provides the strongest robustness in the classification severity ablation, while FedProx also shows relatively stable behavior.

---

# 3. ISIC-2018 Segmentation

The third experiment extends the study from classification to **binary skin-lesion segmentation**.

### Dataset

ISIC-2018 Challenge Task 1:

- 2,595 image-mask pairs
- Binary segmentation
- White pixels represent lesion regions
- Black pixels represent background

### Model

A U-Net architecture with a **ResNet-18 encoder** is used.

When available, the implementation uses:

```text
segmentation_models_pytorch
```

with:

```text
U-Net
ResNet-18 encoder
ImageNet pretrained weights
3 input channels
1 output channel
```

A custom U-Net implementation is also provided as a fallback.

### Training configuration

- Image size: `256 × 256`
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `1e-4`
- Weight decay: `1e-5`
- Loss: Dice + BCE
- Primary metric: Dice

---

# Novel Non-IID Partitioning for Segmentation

Unlike the classification experiments, segmentation clients are not partitioned primarily according to diagnosis.

Instead, client distributions are constructed using **lesion area**.

For every segmentation mask, the fraction of pixels belonging to the lesion is calculated.

The area distribution is divided using:

- 33rd percentile: `0.0715`
- 66th percentile: `0.2416`

This produces three client distributions:

| Client   | Lesion Distribution | Samples |
| -------- | ------------------- | ------: |
| Client 0 | Small lesions       |     688 |
| Client 1 | Medium lesions      |     687 |
| Client 2 | Large lesions       |     700 |

This creates a different type of non-IID heterogeneity from the classification experiments.

The approach is intended to test whether FL behaves differently when clients differ in a **visual/structural characteristic** rather than primarily in class composition.

---

# ISIC-2018 Segmentation Results

| Method          |       Dice |        IoU | Sensitivity | Specificity | Pixel Accuracy |
| --------------- | ---------: | ---------: | ----------: | ----------: | -------------: |
| Centralized     | **0.8860** | **0.8147** |  **0.9066** |      0.9741 |     **0.9538** |
| Best Local      |     0.8482 |     0.7536 |  **0.9200** |      0.9448 |         0.9377 |
| FedAvg          | **0.8742** | **0.8019** |      0.8772 |      0.9754 |     **0.9504** |
| Weighted FedAvg |     0.8741 |     0.8018 |      0.8711 |      0.9777 |         0.9509 |
| FedProx         |     0.8738 |     0.8005 |      0.8647 |  **0.9778** |         0.9499 |
| Ditto           |     0.8703 |     0.7979 |      0.8754 |      0.9729 |         0.9487 |

### Key finding

The federated-to-centralized gap is only approximately:

**0.012 Dice**

This is the smallest centralized-to-federated performance gap observed across the three experiments.

Unlike the classification experiments, all three clients benefit from federation.

Per-client Dice improvements include:

- Client 0: **+0.015**
- Client 1: **+0.128**
- Client 2: **+0.104**

This suggests that segmentation features can generalize across the lesion-area distributions more effectively than class-specific classification features transfer across highly specialized clients.

---

# Explainability Analysis

Explainability is evaluated using **Grad-CAM** and intermediate federated checkpoints.

The project includes three main explainability analyses.

## 1. Real Client Grad-CAM

Grad-CAM is evaluated using actual client checkpoints rather than simulated client models.

For HAM10000, the FedAvg analysis uses 50 held-out test images.

Reported average Grad-CAM IoU:

| Client   | Grad-CAM IoU |
| -------- | -----------: |
| Client 0 |        0.800 |
| Client 1 |        0.900 |
| Client 2 |        0.909 |
| Average  |    **0.870** |

## 2. CAM Evolution

Intermediate global checkpoints are saved at:

```text
Round 1
Round 5
Round 10
Round 15
Round 20
```

This allows the evolution of model attention during federated training to be visualized.

The same test image can be compared across communication rounds and across aggregation strategies.

## 3. Per-Method / Per-Class CAM Comparison

The experiments also compare Grad-CAM outputs for the different aggregation strategies on the same test examples.

This provides qualitative evidence of whether different FL strategies learn different visual attention patterns.

---

# Machine Unlearning

Machine unlearning is evaluated by attempting to remove the contribution of a designated client after federated training.

Three approaches are compared:

### 1. Exact Retraining

The model is retrained using only the retained clients.

This serves as the gold-standard forgetting reference.

### 2. Gradient Ascent

Gradient ascent is performed on the forget-client data, followed by fine-tuning on the retained clients.

### 3. Fine-tuning Only

The model is fine-tuned using the retained clients without explicit gradient ascent.

---

# Segmentation Unlearning Experiment

For ISIC-2018 segmentation:

- Forget client: **Client 2**
- Forgotten distribution: **large lesions**
- Retained clients: Clients 0 and 1
- Gradient ascent steps: 50
- Fine-tuning epochs: 3
- Unlearning learning rate: `5e-5`

### Results

| Method           | Forget Dice | Retain Dice | Forgetting Score | Retention Score |
| ---------------- | ----------: | ----------: | ---------------: | --------------: |
| Original         |       0.906 |       0.925 |            0.000 |           1.000 |
| Exact Unlearning |       0.854 |       0.915 |       **+0.053** |           0.990 |
| Gradient Ascent  |       0.912 |       0.970 |           −0.005 |           1.049 |
| Fine-tune Only   |       0.917 |       0.969 |           −0.011 |           1.048 |

### Finding

Only exact retraining produces actual forgetting in this segmentation experiment.

Gradient ascent and fine-tuning instead improve performance on the forgotten large-lesion client.

This suggests that when the forgotten distribution is defined by lesion size, segmentation features learned from the retained small- and medium-lesion distributions can still generalize to large lesions.

Therefore, the effectiveness of unlearning can depend on the relationship between the forgotten and retained data distributions.

---

# Communication Cost

Communication cost is measured across the federated training process.

For the classification experiments:

| Method          | Communication Cost |
| --------------- | -----------------: |
| FedAvg          |            5.37 GB |
| Weighted FedAvg |            5.37 GB |
| FedProx         |            5.37 GB |
| Ditto           |           10.73 GB |

Ditto requires approximately twice the communication because it maintains both global and personalized model updates.

For segmentation:

| Method          | Communication Cost |
| --------------- | -----------------: |
| FedAvg          |            6.88 GB |
| Weighted FedAvg |            6.88 GB |
| FedProx         |            6.88 GB |
| Ditto           |           13.76 GB |

The higher segmentation cost results from the larger U-Net model.

---

# Cross-Experiment Findings

The experiments reveal several important patterns.

## 1. Non-IID heterogeneity type matters

Weighted FedAvg performs poorly when client dataset size is correlated with class composition.

On ISIC-2019:

- Macro F1 degradation: **−28.6%**
- G-Mean degradation: **−47.5%**

However, Weighted FedAvg performs competitively in the segmentation setting where client sizes are approximately balanced.

This suggests that FL performance cannot be understood solely from the presence of non-IID data. The **structure of the heterogeneity** also matters.

---

## 2. Ditto is robust for classification, but not universally optimal

Ditto achieves the strongest robustness to increasing non-IID severity in the ISIC-2019 classification experiment.

Its macro-F1 degradation from mild to extreme heterogeneity is only:

**−6.4%**

However, Ditto is the lowest-performing method by Dice in the ISIC-2018 segmentation experiment.

This indicates that personalization is not automatically beneficial.

---

## 3. Classification and segmentation respond differently to federation

In classification, specialist clients can experience losses on their majority classes after global aggregation.

In segmentation, all three lesion-area clients benefit from federation.

This suggests that spatial segmentation representations can transfer across lesion-size distributions more effectively than class-specific classification representations transfer across highly specialized clients.

---

## 4. Federated-to-centralized gap depends on the task

| Experiment             | Best FL vs Centralized |
| ---------------------- | ---------------------: |
| HAM10000               |        −0.066 Macro F1 |
| ISIC-2019              |        −0.093 Macro F1 |
| ISIC-2018 Segmentation |            −0.012 Dice |

The segmentation experiment shows the smallest gap between federated and centralized learning.

---

## 5. Machine unlearning depends on distribution structure

The ISIC-2018 unlearning experiment demonstrates that removing a client's training distribution is not necessarily equivalent to degrading performance on that client's distribution.

When the forgotten and retained distributions share transferable visual characteristics, fine-tuning or gradient-based unlearning can fail to produce meaningful forgetting.

Exact retraining remains the reference approach for actual removal in this experiment.

---

# Repository Structure

```text
federated-skin-lesion-classification/
│
├── README.md
├── LICENSE
│
├── HAM10000/
│   └── FL_HAM10000_Classification.ipynb
│
├── ISIC-2019/
│   ├── FL_ISIC2019_Training_FedAvg_WeightedFedAvg.ipynb
│   ├── FL_ISIC2019_Training_FedProx_Ditto.ipynb
│   └── FL_ISIC2019_Validation.ipynb
│
└── ISIC-2018-Segmentation/
    ├── FL_ISIC2018_Training_FedAvg_WeightedFedAvg.ipynb
    ├── FL_ISIC2018_Training_FedProx.ipynb
    ├── FL_ISIC2018_Training_Ditto.ipynb
    └── FL_ISIC2018_Validation.ipynb
```

Each dataset folder contains the notebooks required to reproduce its corresponding experiments.

---

# Notebook Organization

## HAM10000

`FL_HAM10000_Classification.ipynb`

Contains:

- Dataset preparation
- Client partitioning
- Federated training
- FedAvg
- Weighted FedAvg
- FedProx
- Ditto
- Non-IID severity ablation
- Model evaluation
- Communication analysis
- Machine unlearning
- Grad-CAM analysis
- CAM evolution
- Per-class and per-method comparisons

---

## ISIC-2019

### `FL_ISIC2019_Training_FedAvg_WeightedFedAvg.ipynb`

Contains training for:

- FedAvg
- Weighted FedAvg

### `FL_ISIC2019_Training_FedProx_Ditto.ipynb`

Contains training for:

- FedProx
- Ditto

The training was divided across notebooks because of the computational constraints of the Kaggle environment.

### `FL_ISIC2019_Validation.ipynb`

Contains:

- Final evaluation
- Comparison of all four methods
- Ablation analysis
- Communication analysis
- Machine unlearning
- Grad-CAM
- CAM evolution
- Per-class/per-method analysis

---

## ISIC-2018 Segmentation

### `FL_ISIC2018_Training_FedAvg_WeightedFedAvg.ipynb`

Contains training for:

- FedAvg
- Weighted FedAvg

### `FL_ISIC2018_Training_FedProx.ipynb`

Contains:

- FedProx training
- Checkpoint generation
- Validation tracking
- Model saving

### `FL_ISIC2018_Training_Ditto.ipynb`

Contains:

- Ditto training
- Global and personalized model training
- Checkpoint generation
- Validation tracking
- Model saving

### `FL_ISIC2018_Validation.ipynb`

Contains:

- Final segmentation evaluation
- Client-level evaluation
- Mask visualization
- Machine unlearning
- Unlearning comparison
- Communication analysis
- Segmentation analysis

---

# Experimental Reproducibility

The experiments were implemented in **PyTorch** and executed using **Kaggle GPU environments**.

Common experimental settings include:

- Random seed: `42`
- 3 federated clients
- 20 main communication rounds
- 3 local epochs
- AdamW optimization
- Validation-based model selection
- Strictly held-out test data
- Programmatic checks for client partition overlap

The notebooks contain the complete implementation of the corresponding experiments.

Large datasets and generated model checkpoints are not included in this repository.

---

# Limitations

This project is an experimental research study and is **not a clinically validated medical diagnostic system**.

Important limitations include:

- Experiments use simulated federated clients rather than independent real hospitals.
- The number of clients is limited to three.
- Non-IID distributions are constructed artificially to study controlled heterogeneity.
- Client-specific noise is simulated rather than derived from real acquisition differences.
- Computational constraints limit the number of federated rounds and clients.
- The unlearning experiments evaluate specific post-training procedures rather than providing a formal guarantee of privacy removal.
- Grad-CAM provides qualitative/approximate model explanations and should not be interpreted as clinical evidence.
- Results should not be interpreted as evidence that any particular FL strategy is universally optimal.

---

# Technologies

- Python
- PyTorch
- Torchvision
- scikit-learn
- TorchMetrics
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Pillow
- segmentation-models-pytorch
- Kaggle / KaggleHub

---

# Summary

This project studies federated learning for skin-lesion analysis across both **classification and segmentation**.

The experiments demonstrate that:

1. **The type of non-IID heterogeneity strongly affects FL performance.**
2. **Weighted FedAvg can become highly vulnerable to size-correlated class imbalance.**
3. **Ditto provides strong robustness under severe classification heterogeneity but is not universally optimal.**
4. **Segmentation can benefit from federation across lesion-size distributions, producing a very small federated-to-centralized performance gap.**
5. **Federation can improve rare-class performance while simultaneously reducing performance for some specialist classes.**
6. **Explainability analysis can be used to examine how federated models' attention evolves during training.**
7. **Machine unlearning effectiveness depends on the relationship between forgotten and retained data distributions.**
8. **Exact retraining remains the effective forgetting reference in the investigated segmentation setting.**
9. **Personalization introduces an additional communication cost that should be considered alongside performance.**

Overall, the results support the conclusion that:

> **There is no universally optimal federated learning strategy for medical imaging. The effectiveness of an aggregation method depends on the task, the structure of client heterogeneity, and the severity of distributional differences across clients.**
