# Deep Learning-Based Building Damage Classification from Post-Disaster Satellite Images

## Overview

This project is a coursework submission for Data Warehousing and Mining focused on classifying building damage from post-disaster satellite imagery. The implementation uses notebook-based experiments in Google Colab and PyTorch to prepare an image dataset, handle class imbalance, and compare multiple deep learning models for four damage categories.

## Problem Statement

After a disaster, manually checking building damage from satellite imagery is slow and difficult to scale. This project studies whether deep learning models can classify cropped building images into four damage levels:

- `destroyed`
- `major-damage`
- `minor-damage`
- `no-damage`

The project also examines how class imbalance affects model performance by comparing results on both imbalanced and balanced versions of the dataset.

## Core Concepts Used

- Image preprocessing from labeled annotation files
- Bounding-box extraction from polygon annotations
- Class-wise dataset balancing by random undersampling
- Train and validation split creation
- Deep learning model training with PyTorch
- Transfer learning with ResNet50
- Model comparison using accuracy, precision, recall, macro F1 score, classification report, and confusion matrix

## Approach / Workflow

The project is organized as a sequence of notebooks:

1. A preprocessing notebook reads post-disaster annotation files and satellite images, extracts building regions using polygon coordinates, resizes the cropped images, and stores them class-wise.
2. The same notebook creates:
   - an imbalanced dataset that keeps the natural class distribution
   - a balanced dataset created by undersampling all classes to the minimum class count
3. Both datasets are split into `train` and `val` folders using an 80:20 split.
4. Separate model notebooks train and evaluate:
   - a custom CNN
   - a pretrained ResNet50 model
   - a hybrid CNN + ResNet50 model
5. The notebooks print epoch-wise training progress and final validation metrics.

### Recorded Dataset Preparation Details

The preprocessing notebook shows the following assumptions and outputs:

- Source folders expected: `train/images` and `train/labels`
- Only the first `200` `post_disaster` label files are processed
- Damage classes used: `no-damage`, `minor-damage`, `major-damage`, `destroyed`
- Original cropped image counts:
  - `no-damage`: `5258`
  - `minor-damage`: `845`
  - `major-damage`: `1091`
  - `destroyed`: `646`
- Balanced dataset count per class after undersampling: `646`
- Balanced train/validation split:
  - train: `516` images per class
  - val: `130` images per class
- Imbalanced train/validation split:
  - train: `4206`, `676`, `872`, `516`
  - val: `1052`, `169`, `219`, `130`

## Key Features

- Converts annotation-based disaster imagery into cropped building-level classification data
- Preserves both imbalanced and balanced dataset versions for comparison
- Compares three model families: CNN, ResNet50, and hybrid CNN + ResNet50
- Stores recorded notebook outputs including classification reports and confusion matrices
- Includes report and workflow diagram files for academic documentation

## Tech Stack

- Python
- Jupyter Notebook / Google Colab
- PyTorch
- Torchvision
- scikit-learn
- OpenCV
- Shapely
- tqdm

## Project Structure

```text
post-disaster-building-damage-classification/
│── notebooks/
│   ├── 01_dataset_preparation_and_balancing.ipynb
│   ├── 02_cnn_imbalanced.ipynb
│   ├── 03_cnn_balanced.ipynb
│   ├── 04_resnet50_imbalanced.ipynb
│   ├── 05_resnet50_balanced.ipynb
│   ├── 06_hybrid_cnn_resnet_imbalanced.ipynb
│   ├── 07_hybrid_cnn_resnet_balanced.ipynb
│   └── archive/
│       └── phase2_resnet_workbook.ipynb
│── docs/
│   ├── final-report-source.docx
│   ├── project-report.pdf
│   ├── workflow-diagram.drawio
│   ├── workflow-diagram.png
│   └── workflow-diagram.svg
│── README.md
│── requirements.txt
│── .gitignore
```

## How to Run

### Recommended Environment

The notebooks were originally created in Google Colab and use Google Drive mounting plus `/content/...` paths. Running in Colab is the closest match to the original setup.

### Option 1: Run in Google Colab

1. Upload the repository to Google Drive or GitHub.
2. Open the required notebook in Colab.
3. Mount Google Drive when prompted by the notebook.
4. Place the input dataset in the folder structure expected by the notebook.
5. Run the notebooks in this order:
   - `01_dataset_preparation_and_balancing.ipynb`
   - any one or more of the model notebooks

### Option 2: Run Locally

1. Create a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Jupyter:

```bash
jupyter notebook
```

4. Open the notebook you want to run.
5. Replace Google Drive mount cells and `/content/...` paths with local paths before execution.

### Notebook Execution Order

- `01_dataset_preparation_and_balancing.ipynb`
- `02_cnn_imbalanced.ipynb`
- `03_cnn_balanced.ipynb`
- `04_resnet50_imbalanced.ipynb`
- `05_resnet50_balanced.ipynb`
- `06_hybrid_cnn_resnet_imbalanced.ipynb`
- `07_hybrid_cnn_resnet_balanced.ipynb`

## Dataset / Input

The project expects an xView2 or xBD-style post-disaster dataset layout with image and label files. The raw dataset is **not included** in this repository.

Expected source structure used by the preprocessing notebook:

```text
train/
├── images/
└── labels/
```

Important notes:

- The preprocessing notebook reads JSON labels and corresponding PNG images.
- It only processes the first `200` `post_disaster` label files found in the label directory.
- The generated dataset is then organized into class-wise folders for model training.

## Output / Results

The notebooks contain recorded results from earlier executions. These results were preserved from the original project files and were **not re-run during repository cleanup** because the dataset is not bundled with the project.

### Recorded Final Metrics from Notebook Outputs

| Model Notebook | Data Version | Accuracy | Precision | Recall | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| CNN | Imbalanced | 0.7045 | 0.4572 | 0.4106 | 0.3802 |
| CNN | Balanced | 0.5404 | 0.6185 | 0.5404 | 0.5126 |
| ResNet50 | Imbalanced | 0.8478 | 0.7613 | 0.7808 | 0.7703 |
| ResNet50 | Balanced | 0.7058 | 0.7043 | 0.7058 | 0.7000 |
| Hybrid CNN + ResNet50 | Imbalanced | 0.8478 | 0.7579 | 0.7792 | 0.7661 |
| Hybrid CNN + ResNet50 | Balanced | 0.7212 | 0.7233 | 0.7212 | 0.7193 |

### Result Notes

- The written report mainly discusses the CNN and ResNet50 experiments.
- The hybrid notebooks are preserved as additional experiments because they are present in the project files and contain recorded outputs.
- On the documented CNN and ResNet50 comparison, the balanced ResNet50 configuration gives a stronger trade-off between overall accuracy and class-balanced performance than the balanced CNN baseline.

## What This Project Demonstrates

- How labeled satellite imagery can be converted into a building-level classification dataset
- Why class imbalance matters in multiclass disaster-damage prediction
- The difference between a custom CNN baseline and transfer learning with ResNet50
- How balancing a dataset can change macro F1 and class-wise performance interpretation
- How notebook-based experiments can be used to compare multiple model configurations

## Current Limitations

- The raw dataset is not included in the repository.
- The notebooks are tightly coupled to Google Colab and Google Drive paths.
- There is no standalone Python package or training script.
- The preprocessing notebook uses only the first `200` `post_disaster` label files, so this is a reduced-scope dataset sample.
- There is no separate test set; the notebooks use only train and validation splits.
- Trained model weight files are not included.
- The hybrid experiments are present in notebooks but are not clearly documented in the main written report.
- The project does not include automated tests, experiment tracking, or reproducible environment pinning.

## Future Improvements

- Convert the notebook workflow into reusable Python scripts with configurable paths.
- Add a clear dataset preparation guide or manifest for the expected input folders.
- Include a dedicated test split for final model comparison.
- Save trained model checkpoints and sample prediction outputs in a structured way.
- Compare balancing methods such as weighted loss or oversampling in addition to undersampling.
- Add training curves and confusion matrix figures as exported outputs for easier review.

## AI Assistance Disclosure

“This project was developed as part of academic coursework. AI tools (such as ChatGPT) were used for assistance in structuring, debugging, and documentation. The implementation, logic, and understanding were developed and verified independently.”

## My Contribution / What I Learned

- I worked on understanding how disaster image annotations can be converted into building-level training samples.
- I learned how class imbalance can make raw accuracy misleading in multiclass classification problems.
- I practiced comparing a simple CNN baseline with a transfer-learning model such as ResNet50.
- I gained experience reading classification reports, confusion matrices, and macro F1 scores instead of relying on a single metric.
- I also learned the importance of documenting assumptions such as dataset sampling, folder structure, and notebook-specific setup.
