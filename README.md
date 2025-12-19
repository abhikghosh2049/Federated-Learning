
# 🌼 Federated Learning Simulation with Flower & PyTorch

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flower](https://img.shields.io/badge/Flower-Federated_Learning-orange?style=for-the-badge)
![Colab](https://img.shields.io/badge/Google_Colab-Compatible-F9AB00?style=for-the-badge&logo=googlecolab)

A simulation of a **Federated Learning (FL)** system where **10 clients** collaboratively train a Convolutional Neural Network (CNN) on the **CIFAR-10** dataset without sharing their raw data. This project uses the **Flower (`flwr`)** framework to orchestrate the distributed training process.

---

## 🚀 Project Overview

Traditional machine learning requires collecting data on a central server. **Federated Learning** enables devices (clients) to train a model locally and share only the model updates (weights), preserving data privacy.

This project demonstrates:
* **Data Partitioning:** Splitting CIFAR-10 into 10 unique partitions (one per client) using `flwr-datasets`.
* **Privacy-Preserving Training:** Raw data never leaves the client.
* **Federated Averaging:** Using the `FedAvg` strategy to aggregate model updates on the server.
* **Simulation:** Running the entire Client-Server architecture within a single environment using Flower's simulation engine.

## 🏗️ Architecture



[Image of Federated Learning Architecture]


The system consists of three main components:
1.  **The Server:** Orchestrates the rounds, aggregates weights using `FedAvg`, and evaluates the global model.
2.  **The Clients (x10):** Each client holds a unique slice of the dataset. They receive the global model, train it for 1 epoch locally, and return the updated weights.
3.  **The Model:** A custom PyTorch CNN with 2 convolutional layers and 3 fully connected layers.

## 🛠️ Installation & Setup

This project handles specific dependency conflicts often found in Google Colab (specifically regarding `protobuf` and `grpcio`).

### Prerequisites
* Python 3.7+
* Jupyter Notebook or Google Colab

### Dependencies
Run the following commands to set up the environment (as seen in the notebook):

```bash
# Uninstall conflicting versions first
pip uninstall -y protobuf grpcio

# Install Flower and PyTorch
pip install -q "flwr[simulation]" flwr-datasets[vision] torch torchvision matplotlib

# Install compatible protocol buffer versions
pip install -U "protobuf==4.25.3" "grpcio==1.60.0"
