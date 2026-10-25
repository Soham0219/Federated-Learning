# Federated Learning 

A privacy preserving approach to training a next word prediction engine using decentralized model optimization and privacy safeguards, ensuring user text data never leaves edge devices.

---

## Table of Contents
* [Project Concept](#project-concept)
* [The Main Idea](#the-main-idea)
* [System Design](#system-design)
* [Model Architecture](#model-architecture)
* [Core Features](#core-features)
* [Project Layout](#project-layout)
* [Setup and Installation](#setup-and-installation)
* [How to Run](#how-to-run)
* [Hyperparameter Matrix](#hyperparameter-matrix)
* [Expected Console Logs](#expected-console-logs)
* [Advanced Scenarios](#advanced-scenarios)
* [Academic References](#academic-references)

---

## Project Concept

Modern predictive text engines process highly sensitive personal messages, chat histories, and private credentials. Sending this huge volume of raw typing data to a centralized server creates high security risks, privacy issues, and compliance complications. 

This repository provides a framework simulating a decentralized environment where multiple separate devices (clients) work together to train a multi-layer Long Short-Term Memory (LSTM) network without sharing their local text files.

---

## The Main Idea

### Traditional Centralized Workflows
```
Node A Data ---┐
Node B Data ---┼-> Central Server Warehouse ---> Model Training ---> Production Push
Node C Data ---┘     (High Security Risk)
```
Traditional machine learning collects all data in one central hub. A single security breach compromises the entire user base, exposing private information to leaks and regulatory failures like GDPR or CCPA.

### Federated Training Workflow
```
┌───────────────┐        Global Model Weights    ┌────────────────────┐
│ Edge Client 1 │ ◄────────────────────────────► │                    │
│ (Local Text)  │        Updated Local Weights   │                    │
├───────────────┤ ─────────────────────────────► │ Aggregation Server │
│ Edge Client 2 │                                │                    │
│ (Local Text)  │ ◄────────────────────────────► │ * Runs FedAvg      │
├───────────────┤                                │ * Adds DP Noise    │
│ Edge Client N │ ─────────────────────────────► │ * NEVER Sees Data  │
│ (Local Text)  │                                │                    │
└───────────────┘                                └────────────────────┘
```
In this framework, data remains safe on the local device. The process works like this:
1. The main server sends the current global model state to participating client devices.
2. Each client trains the model locally using its own on-device text data.
3. Clients strip personal identifiers and return only the calculated weight updates to the server.
4. The server combines these updates into a new global model using the FedAvg protocol.
5. The cycle repeats until the model achieves good performance.

---

## System Design

### The Learning Loop
```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING ROUND t                         │
│                                                                 │
│  1. Server sends active global model weights to K clients.      │
│  2. Target Client Devices:                                      │
│     a. Set local model equal to the global weights.             │
│     b. Run training loops on local text for E epochs.           │
│     c. Send resulting weight matrices back to the server.       │
│  3. Server Consolidation:                                       │
│     W_new = sum (Samples_Client / Samples_Total) * W_Client     │
│  4. Evaluate global model performance on a clean test set.     │
│  5. Early stopping stops training if test loss stops improving. │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Architecture

The model uses a standard multi-layer Recurrent Network built for sequence language prediction:

```
Raw Input Text ("I am looking forward")
               │
               ▼
┌──────────────────────────────┐
│       Embedding Layer        │  Maps words into dense 64-dimensional vectors
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│      LSTM Layer 1 (128)      │  Processes sequence text with recurrent dropout
│   LayerNorm and Dropout      │  Stabilizes gradient flow during training
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│      LSTM Layer 2 (128)      │  Captures structural, long-range text patterns
│   LayerNorm and Dropout      │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│     Dense Project Layer      │  Transforms features using ReLU activation
│   LayerNorm and Dropout      │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│    Softmax Output Layer      │  Calculates probabilities over the vocabulary
└──────────────┬───────────────┘
               ▼
   Target Word Prediction ("to")
```

---

## Core Features

* **Decentralized Training:** Uses the standard Federated Averaging framework out of the box.
* **Non-IID Drift Protection:** Includes the FedProx penalty parameter (mu) to keep local models from drifting too far apart when users write in completely different styles.
* **Differential Privacy:** Supports adding Gaussian noise during server aggregation for formal privacy protection.
* **Stable Training:** Uses Layer Normalization layers between LSTM stacks to prevent exploding or vanishing gradient issues.
* **Early Stopping:** Automatically stops training if the test loss stops improving for 7 rounds in a row.
* **Flexible Data Handling:** Can load data from built-in sentence lists, custom text files, or automatically download Shakespeare texts.

---

## Project Layout

```
Federated Learning
 ├── fl_main.py         # Main training framework, models, and server logic
 ├── requirements.txt   # System package configurations
 └── README.md          # Project documentation
```

---

## Setup and Installation

### System Requirements
* Python 3.9, 3.10, or 3.11
* The pip package manager

### 1. Download the Project
```bash
git clone <your-repository-url>
cd Federated-Learning
```

### 2. Create a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS or Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

## How to Run

### Standard Run
Launch the standard federated learning simulation using default settings:
```bash
python fl_main.py
```

### Changing Run Settings
You can modify configurations directly inside the code at the bottom of the `fl_main.py` file:
```python
cfg = FLConfig(
    num_clients=6,          # Number of participating client devices
    num_rounds=40,          # Maximum number of training rounds
    fraction_clients=0.5,   # Ratio of clients selected per round
    sequence_length=4,      # Context word window size
    local_epochs=3,         # Local training epochs per client per round
    lstm_units=128,         # Number of hidden units in LSTM layers
    data_source="builtin"   # Can be "builtin", "shakespeare", or a text file path
)
federated_learning(cfg)
```

---

## Hyperparameter Matrix

| Configuration Variable | Default Value | Description |
| :--- | :--- | :--- |
| `embedding_dim` | `64` | Vector size for word tracking. |
| `lstm_units` | `128` | Hidden layer width for processing context. |
| `num_lstm_layers` | `2` | Number of stacked LSTM layers. |
| `dropout_rate` | `0.3` | Dropout percentage used for regularization. |
| `sequence_length` | `4` | Number of previous words used as context. |
| `learning_rate` | `0.001` | Learning rate for the optimization process. |
| `mu` | `0.0` | FedProx penalty coefficient to manage client drift. |
| `noise_multiplier`| `0.0` | Gaussian noise scale factor for differential privacy. |

---

## Expected Console Logs

```
=================================================================
  Federated Learning — Next Word Prediction (LSTM)
=================================================================
  Clients        : 3
  Rounds         : 50
  Selection frac : 1.0
  Sequence len   : 3
  Vocab size     : 315
  Train sentences: 178
=================================================================
  Round   1/50  |  loss 5.6214  |  acc 0.0125  |  ppl 276.28
  Round   2/50  |  loss 5.1042  |  acc 0.0541  |  ppl 164.71
  ...
  Round  34/50  |  loss 2.9103  |  acc 0.2245  |  ppl 18.36
  
  Early stopping at round 34 (no improvement for 7 rounds)

=================================================================
  Final Results
=================================================================
  Test loss       : 2.9103
  Test accuracy   : 0.2245
  Test perplexity : 18.36

  Next-word predictions:
    "I am going"  →  to (44.12%), the (9.05%), out (4.21%)
    "She is"      →  reading (14.20%), running (8.62%), studying (7.11%)
=================================================================
```

---

## Advanced Scenarios

### Handling Non-IID Local Data
If different client devices have completely different writing styles, enable the proximal penalty term to keep the training stable:
```python
cfg = FLConfig(mu=0.01)  # Enables the FedProx regularization penalty
```

### Adding Differential Privacy
To mathematically protect the model weights and ensure no private user text can be reconstructed, add noise during the server merge step:
```python
cfg = FLConfig(
    noise_multiplier=0.15,  # Scale factor for DP noise injection
    l2_norm_clip=1.0        # Max gradient clip threshold
)
```

---

## Academic References

1. **McMahan et al.** - *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017. (Introduced the foundational FedAvg algorithm).
2. **Li et al.** - *Federated Optimization in Heterogeneous Networks*, MLSys 2020. (Introduced the FedProx framework to handle non-IID settings).
3. **Abadi et al.** - *Deep Learning with Differential Privacy*, ACM CCS 2016. (Established the groundwork for differential privacy in deep networks).

---

<p align="center">
  Built with TensorFlow - Federated by Design - Privacy First
</p>