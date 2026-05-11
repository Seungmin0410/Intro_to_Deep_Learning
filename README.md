# Intro to Deep Learning - class projects

---

## Projects & Features

### 1. k-Nearest Neighbors (kNN) from Scratch
* **File:** `01_knn_mnist.py`
* **Description:** Implementation of the kNN classification algorithm using only NumPy to understand the fundamentals of distance-based learning.
* **Features:** * Manual L2 distance calculation for vector operations.
    * MNIST digit classification with hyperparameter $k$ tuning.
    * Evaluates classification accuracy and execution time.

### 2. LeNet-5 Architecture Analysis (CIFAR-10)
* **Files:** `02_LeNet5_baseline.py`, `03_LeNet5_batchnormal.py`, `04_LeNet5_dropout.py`
* **Description:** A series of experiments on the classic LeNet-5 architecture applied to the CIFAR-10 dataset to explore various regularization methods.
* **Key Implementations:**
    * **Baseline:** Standard CNN implementation with PyTorch.
    * **Batch Normalization:** Applied `BatchNorm2d` to stabilize and accelerate the training process.
    * **Dropout:** Integrated `Dropout` layer (p=0.2) to prevent overfitting and improve generalization.

### 3. Final Project: Model Compression via Pruning
* **File:** `05_final_project_pruning.py`
* **Description:** Project on optimization using weight pruning techniques for efficient AI.
* **Features:**
    * **L1 Unstructured Pruning:** Removed 70% of network weights using `torch.nn.utils.prune`.
    * **Fine-tuning:** Recovered model accuracy through post-pruning training cycles.
    * **Analysis:** Calculated total/pruned parameter counts and final pruning ratio for efficiency verification.
