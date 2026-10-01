# NetraX

## AI-Powered Retinal Disease Analysis & Decision Support

NetraX is an AI-powered retinal image analysis platform for multi-label analysis of fundus photographs. It combines a React frontend, FastAPI backend, PostgreSQL persistence, and a TensorFlow/Keras EfficientNetB0 model to generate predictions across 45 retinal conditions.

> **Medical Disclaimer:** NetraX is a research and decision-support system. Its predictions are not medical diagnoses and should not be used as a substitute for examination, diagnosis, or treatment by a qualified medical professional.

---

## Overview

NetraX provides an end-to-end workflow for authenticated retinal image analysis:

```text
User
 |
 v
React Frontend
 |
 | Fundus Image Upload
 v
FastAPI Backend
 |
 +-- JWT Authentication
 +-- Image Validation
 +-- ML Inference
        |
        v
   EfficientNetB0
        |
        v
  45 Sigmoid Outputs
        |
        v
Per-Label Thresholds
        |
        +-- Probability
        +-- Confidence
        +-- Detection Status
        |
        v
PostgreSQL
Analysis History
```

---

## Features

- Fundus image upload and preview
- JWT-based authentication
- Multi-label retinal disease prediction
- 45 retinal condition classes
- EfficientNetB0 deep learning model
- Validation-derived per-label thresholds
- Prediction probabilities and confidence values
- Detection status for each condition
- Analysis history
- PostgreSQL persistence
- FastAPI REST API
- React + Vite frontend
- Responsive dashboard
- Decision-support focused UI
- Explicit medical-use limitations

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, Vite, Tailwind CSS, React Router, Axios, Lucide React, Recharts |
| Backend | Python, FastAPI, Uvicorn, SQLAlchemy, PostgreSQL |
| Authentication | JWT, bcrypt |
| Machine Learning | TensorFlow, Keras, EfficientNetB0 |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Datasets | RFMiD, ODIR-5K |

---

# System Architecture

```text
                         +---------------------+
                         |       User          |
                         +----------+----------+
                                    |
                                    v
                         +---------------------+
                         |   React Frontend    |
                         |  Vite + Tailwind    |
                         +----------+----------+
                                    |
                              REST / JWT
                                    |
                                    v
                         +---------------------+
                         |   FastAPI Backend   |
                         |                     |
                         | Authentication      |
                         | Image Validation    |
                         | Inference Service   |
                         +-------+-------+-----+
                                 |       |
                    +------------+       +------------+
                    v                                 v
          +-----------------+               +-----------------+
          | EfficientNetB0  |               |   PostgreSQL    |
          |   45 Outputs    |               | Users/Analyses  |
          +--------+--------+               +-----------------+
                   |
                   v
          +-----------------+
          | Threshold-Based |
          |   Predictions   |
          +-----------------+
```

---

# Machine Learning

## Model Architecture

NetraX uses an ImageNet-pretrained EfficientNetB0 backbone.

```text
Input Fundus Image
       |
       v
384 x 384 x 3
       |
       v
EfficientNetB0
       |
       v
Global Average Pooling
       |
       v
Batch Normalization
       |
       v
Dropout
       |
       v
Dense 256 ReLU
       |
       v
Dropout
       |
       v
45 Sigmoid Outputs
```

### Production Model

| Property | Value |
|---|---|
| Architecture | EfficientNetB0 |
| Input | 384 x 384 x 3 |
| Output | 45 sigmoid probabilities |
| Parameters | 4,394,192 |
| Production model | `netrax_finetuned_best.keras` |

The trained model is intentionally excluded from Git because of its size.

---

# Retinal Conditions

The model contains 45 output labels:

| # | Label | # | Label | # | Label |
|---:|---|---:|---|---:|---|
| 1 | DR | 16 | ODP | 31 | PRH |
| 2 | ARMD | 17 | ODE | 32 | MNF |
| 3 | MH | 18 | ST | 33 | HR |
| 4 | DN | 19 | AION | 34 | CRAO |
| 5 | MYA | 20 | PT | 35 | TD |
| 6 | BRVO | 21 | RT | 36 | CME |
| 7 | TSLN | 22 | RS | 37 | PTCR |
| 8 | ERM | 23 | CRS | 38 | CF |
| 9 | LS | 24 | EDN | 39 | VH |
| 10 | MS | 25 | RPEC | 40 | MCA |
| 11 | CSR | 26 | MHL | 41 | VS |
| 12 | ODC | 27 | RP | 42 | BRAO |
| 13 | CRVO | 28 | CWS | 43 | PLQ |
| 14 | TV | 29 | CB | 44 | HPED |
| 15 | AH | 30 | ODPM | 45 | CL |

---

# Datasets

## RFMiD

RFMiD was used as the primary labeled dataset and as the untouched final evaluation benchmark.

| Split | Images |
|---|---:|
| Training | 1,920 |
| Validation | 640 |
| Test | 640 |
| **Total** | **3,200** |

RFMiD provides labels for all 45 output classes.

The RFMiD test set was kept separate from training and model development and was used for final evaluation.

---

## ODIR-5K

ODIR-5K was incorporated as an additional training source.

After patient-level splitting and filtering to available mapped eye images:

| Split | Mapped eye images |
|---|---:|
| Training | 1,701 |
| Validation | 205 |
| Test | 199 |
| **Total** | **2,105** |

Conservative mappings from ODIR diagnoses to RFMiD-compatible labels were used for:

- DR
- ARMD
- MYA
- ERM
- DN
- BRVO
- MNF
- TSLN
- LS

Diagnoses that could not be reliably mapped were left unmapped rather than being assigned to an unrelated RFMiD class.

---

# Data Preprocessing

Fundus images are processed using:

```text
Image
  |
  v
RGB Conversion
  |
  v
Resize With Padding
  |
  v
384 x 384
  |
  v
Float32 [0, 255]
  |
  v
Model
```

Training augmentation includes:

- Random brightness adjustment
- Random contrast adjustment
- Horizontal flipping

---

# Handling Partial Labels

A key part of the combined RFMiD + ODIR training pipeline is handling differences in annotation coverage.

RFMiD provides authoritative labels for all 45 classes.

ODIR only provides explicit information for mapped diagnoses. Therefore, an absent ODIR diagnosis was not automatically treated as a negative example.

The V2 pipeline uses a label mask:

```text
RFMiD
 +-- All 45 labels known

ODIR
 +-- Explicit mapped positives -> known
 +-- Other labels             -> unknown / ignored
```

Unknown labels are excluded from the masked weighted binary cross-entropy loss.

This prevents unknown ODIR labels from incorrectly becoming negative training examples.

---

# Training Experiments

NetraX includes three evaluated model configurations.

## V1

The production V1 model uses:

- EfficientNetB0
- ImageNet initialization
- Weighted binary cross-entropy
- Frozen-backbone baseline training
- Fine-tuning of the final EfficientNet layers
- Class weighting
- Validation-derived per-label thresholds

The resulting V1 model is integrated into the FastAPI inference service.

---

## V2

V2 investigated combined RFMiD + ODIR training.

Training data:

```text
RFMiD: 1,920
ODIR:  1,701
----------------
Total: 3,621
```

V2 used:

- EfficientNetB0
- ImageNet initialization
- Masked weighted BCE
- RFMiD validation for primary model selection
- ODIR as an additional training source

---

## V2.1

V2.1 investigated fine-tuning the EfficientNet backbone during combined RFMiD + ODIR training.

It used:

- EfficientNetB0
- Fine-tuning of final backbone layers
- Frozen BatchNorm layers
- Masked weighted BCE
- RFMiD validation
- Validation macro ROC-AUC for model selection

V2.1 was evaluated separately and was not integrated as the production model.

---

# Evaluation

The RFMiD test set contains 640 images and was reserved for final evaluation.

The evaluated metrics were:

- Macro Precision
- Macro Recall
- Macro F1
- Micro Precision
- Micro Recall
- Micro F1
- Mean ROC-AUC
- Mean PR-AUC

## RFMiD Test Results

| Metric | V1 | V2 | V2.1 |
|---|---:|---:|---:|
| Macro Precision | 0.1526 | 0.1657 | 0.1532 |
| Macro Recall | 0.2115 | 0.2431 | 0.2463 |
| Macro F1 | 0.1687 | 0.1755 | 0.1607 |
| Micro Precision | 0.3789 | 0.2705 | 0.2200 |
| Micro Recall | 0.5964 | 0.5710 | 0.5486 |
| Micro F1 | 0.4634 | 0.3671 | 0.3141 |
| Mean ROC-AUC | 0.8111 | 0.8044 | 0.7833 |
| Mean PR-AUC | 0.2435 | 0.2388 | 0.2061 |

The measured evaluation results led to the V1 model being retained for the production inference pipeline. V2 and V2.1 remain documented experimental configurations.

---

# Threshold Calibration

NetraX does not use one universal threshold for all 45 conditions.

Instead, decision thresholds are derived independently from validation data.

The production inference service loads:

```text
ml/evaluation/finetuned_per_label_thresholds.json
```

Each prediction contains:

```text
label
probability
confidence
threshold
detected
```

Example:

```json
{
  "label": "DR",
  "probability": 0.8350,
  "confidence": 83.50,
  "threshold": 0.83,
  "detected": true
}
```

A threshold is a model decision boundary and should not be interpreted as clinical certainty.

---

# Backend

The backend is built with FastAPI.

## API Endpoints

### Authentication

```text
POST /api/auth/register
POST /api/auth/login
```

### Prediction

```text
POST /api/predict
```

Requires authentication and accepts a multipart image upload.

### Analysis History

```text
GET /api/analyses
```

Returns analyses associated with the authenticated user.

### API Documentation

When running locally:

```text
http://127.0.0.1:8000/docs
```

---

# Prediction Workflow

```text
POST /api/predict
       |
       v
JWT Authentication
       |
       v
Image Validation
       |
       v
Image Preprocessing
       |
       v
EfficientNetB0
       |
       v
45 Probabilities
       |
       v
Per-Label Thresholds
       |
       v
Prediction Metadata
       |
       v
PostgreSQL Persistence
       |
       v
JSON Response
```

---

# Database

NetraX uses PostgreSQL for persistent application data.

The database stores:

- User accounts
- Retinal analysis records
- Prediction results associated with analyses

The frontend retrieves authenticated analysis history through the backend API.

---

# Frontend

The frontend is built with React and Vite.

Main application areas:

```text
Landing Page
     |
     v
Authentication
     |
     v
Dashboard
     |
     +-- New Analysis
     |
     +-- Prediction Results
     |
     +-- Analysis History
```

The interface uses an olive, cream, beige, and light-gray visual theme.

The UI explicitly communicates that NetraX provides decision support and does not establish a clinical diagnosis.

---

# Project Structure

```text
NetraX/
|
+-- backend/
|   +-- app/
|   |   +-- core/
|   |   +-- models/
|   |   +-- routes/
|   |   +-- services/
|   +-- tests/
|   +-- openapi.json
|
+-- frontend/
|   +-- src/
|   |   +-- components/
|   |   +-- pages/
|   |   +-- services/
|   +-- package.json
|
+-- ml/
|   +-- dataset/
|   |   +-- build_v2_manifests.py
|   |   +-- fix_v2_manifests.py
|   |
|   +-- evaluation/
|   |   +-- final_test_metrics.csv
|   |   +-- final_test_summary.json
|   |   +-- finetuned_per_label_thresholds.json
|   |   +-- v1_v2_v21_rfmid_comparison.csv
|   |   +-- v2_rfmid_test_metrics.csv
|   |   +-- v21_rfmid_test_metrics.csv
|   |
|   +-- scripts/
|       +-- preprocessing.py
|       +-- dataset.py
|       +-- model.py
|       +-- finetune_model.py
|       +-- train_v2.py
|       +-- evaluate_v2.py
|       +-- evaluate_v21.py
|       +-- v2_dataset.py
|       +-- v2_loss.py
|   
+-- .gitignore
+-- README.md
```

---

# Local Development

## Prerequisites

Install:

- Python
- Node.js
- npm
- PostgreSQL
- Git

## Backend

From the project root:

```powershell
cd backend
python -m venv venv
.env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configure the required environment variables.

Start FastAPI:

```powershell
python -m uvicorn app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## Frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally:

```text
http://localhost:5173
```

## PostgreSQL

Create a PostgreSQL database and configure the backend database connection using the project's environment configuration.

Do not commit database passwords, JWT secrets, API keys, or other credentials.

---

# Machine Learning Artifacts

Large datasets and trained model files are intentionally excluded from Git.

The repository ignores:

```text
ml/models/
ml/dataset/odir_v2/
*.keras
*.h5
*.ckpt
```

The datasets must be obtained separately for training and evaluation.

The repository contains the scripts required to reproduce the dataset preparation and ML experiments.

---

# Reproducibility

The ML directory contains scripts for:

- Image preprocessing
- Dataset creation
- Dataset inspection
- V2 manifest generation
- ODIR mapping preparation
- Masked loss calculation
- V2 training
- V2 evaluation
- V2.1 evaluation
- Threshold generation
- Model comparison

Evaluation results are stored as CSV and JSON artifacts.

---

# Limitations

### Dataset Imbalance

The 45 retinal conditions are highly imbalanced. Some conditions have substantially fewer positive samples than common conditions.

### Rare Conditions

Very rare labels have limited positive examples in the RFMiD test set, making their individual metrics less stable.

### Dataset Mapping

Only conservative ODIR-to-RFMiD mappings were used. Diagnoses without a reliable mapping were not forced into another class.

### Generalization

Performance measured on RFMiD does not establish equivalent performance on other cameras, hospitals, populations, or image acquisition protocols.

### Clinical Validation

NetraX has not been established as a clinically validated diagnostic device.

### Probability Interpretation

Model probabilities represent model outputs. They do not represent disease severity, clinical certainty, or a confirmed diagnosis.

---

# Responsible Use

NetraX is intended for:

- AI/ML research
- Retinal image analysis research
- Educational experimentation
- Computer vision development
- Decision-support research

It should not be used as a standalone clinical diagnostic system.

Potentially concerning results should be evaluated by an appropriately qualified healthcare professional.

---

# Future Work

Potential improvements include:

- Larger and more diverse retinal datasets
- Improved rare-class learning
- Independent external validation
- Better probability calibration
- Explainability using methods such as Grad-CAM
- Image quality assessment
- Domain adaptation
- Clinical validation
- Model versioning
- ML monitoring
- Audit logging
- Cloud deployment
- More robust production infrastructure

---

# Development History

NetraX development involved controlled model experiments:

```text
                    +-----------------+
                    |       V1        |
                    | RFMiD           |
                    | EfficientNetB0  |
                    | Fine-tuning     |
                    +--------+--------+
                             |
                             v
                    Production Model


                    +-----------------+
                    |       V2        |
                    | RFMiD + ODIR    |
                    | Masked BCE      |
                    +--------+--------+
                             |
                             v
                         Evaluation


                    +-----------------+
                    |      V2.1       |
                    | RFMiD + ODIR    |
                    | Fine-tuning     |
                    | Masked BCE      |
                    +--------+--------+
                             |
                             v
                         Evaluation
```

The experiments are retained to document the development and evaluation process.

---

# Project Status

**Status: Working V1 production integration**

The local end-to-end workflow has been tested:

```text
Authentication
      |
      v
Image Upload
      |
      v
FastAPI API
      |
      v
EfficientNetB0 Inference
      |
      v
45 Predictions
      |
      v
Per-Label Thresholding
      |
      v
PostgreSQL Persistence
      |
      v
Frontend Results
      |
      v
Analysis History
```

---

# Author

**Yash Pardeshi**

B.E. — Artificial Intelligence & Data Science

- GitHub: https://github.com/yashpardeshi5514
- LinkedIn: https://www.linkedin.com/in/yashpardeshi5514

---

# Disclaimer

NetraX is an AI research and decision-support project.

**Predictions generated by NetraX are not medical diagnoses.**

The system should not be used as a replacement for professional medical evaluation, diagnosis, or treatment.
