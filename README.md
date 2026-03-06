# Lung & Colon Cancer Histology Classifier (PyTorch / CNN)

This project implements a **convolutional neural network (CNN)** for classifying **histological images** of lung and colon tissue into 5 cancer-related categories. The model is trained using **PyTorch Lightning** and evaluated with a detailed classification report and confusion matrix.

## Classes

The model distinguishes between the following tissue types:

| Label | Description |
|---|---|
| `lung_aca` | Lung adenocarcinoma |
| `lung_scc` | Lung squamous cell carcinoma |
| `lung_n` | Healthy lung tissue |
| `colon_aca` | Colon adenocarcinoma |
| `colon_n` | Healthy colon tissue |

## How It Works

- Raw histology images are split into **train / val / test** sets (80% / 10% / 10%) using `prepare_data.py`.
- The CNN model is defined in `main.py` and consists of **3 convolutional blocks** followed by fully connected layers.
- Training is managed by a **PyTorch Lightning Trainer** with GPU acceleration support.
- After training, `test_model.py` loads the latest checkpoint and generates a **classification report** and **confusion matrix**.

## Project Structure

```
├── prepare_data.py   # Splits raw images into train/val/test folders
├── main.py           # Model definition, data module, and training loop
├── test_model.py     # Loads checkpoint and evaluates model on test set
└── data/
    ├── train/
    ├── val/
    └── test/
```

## Model Architecture

- **Input:** RGB image resized to 256×256
- **Conv Block 1:** Conv2d(3 → 32) + ReLU + MaxPool
- **Conv Block 2:** Conv2d(32 → 64) + ReLU + MaxPool
- **Conv Block 3:** Conv2d(64 → 128) + ReLU + MaxPool
- **FC Layer:** Linear(128×32×32 → 512) + ReLU
- **Output Layer:** Linear(512 → 5)
- **Loss:** Cross-Entropy
- **Optimizer:** Adam (lr=0.001)

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- torchvision
- scikit-learn
- matplotlib
- seaborn
- pandas

Install dependencies:

```bash
pip install torch torchvision pytorch-lightning scikit-learn matplotlib seaborn pandas
```

## Usage

### 1. Prepare the dataset

Edit the `SOURCE_DIR` and `DEST_DIR` paths in `prepare_data.py`, then run:

```bash
python prepare_data.py
```

### 2. Train the model

Edit the `DATA_PATH` in `main.py`, then run:

```bash
python main.py
```

Training checkpoints are saved automatically to the `lightning_logs/` directory.

### 3. Evaluate the model

```bash
python test_model.py
```

This will print a **classification report** (precision, recall, F1-score) and display a **confusion matrix** for the test set.

## Technologies Used

- **Python**
- **PyTorch** & **PyTorch Lightning**
- **torchvision**
- **scikit-learn**
- **Matplotlib / Seaborn**
