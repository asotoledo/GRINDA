# GRINDA - Graph-based Representation for Intelligence and Network Disruption Analysis

> 🚨 An AI-powered framework to identify critical actors (key roles) in covert criminal networks using GNNs, anomaly detection, and disruption simulation.

---

## 📌 Overview

**GRINDA** is a cutting-edge analytical framework designed to uncover strategically important individuals in covert and adaptive criminal networks. It combines advanced **Graph Neural Networks (GNNs)** — including **Graph Attention Networks (GATs)** and **Graph Transformers (GTs)** — with robust **anomaly detection** and **disruption simulation** techniques.

Tested on 51 real criminal networks from official law enforcement datasets in Brazil, GRINDA demonstrates high **scalability**, **generalizability**, and **practical utility** for intelligence-driven decision-making in public security.

---

## 🧠 Key Features

- Transforms raw intelligence data into structured graph representations;
- Learns deep node embeddings using GATs and Graph Transformers;
- Detects anomalous agents using:
  - Adversarial networks (GANs): Graph Attention Network and Graph Transformers;
  - Methods (Isolation Forest, Robust Covariance, SGD One-Class Suport Vector Machine, Outlier Scores, Cosine Autoencoder Detection Agent, Deep Support Vector Data Description, and Anomaly Detection with Generative Adversarial Networks);
- Simulates strategic removals to assess structural impact;
- Modular and scalable pipeline for operational deployment.

---

## 🧪 Real-World Application

GRINDA was validated on 51 real criminal networks using police intelligence datasets (REDS and CVO) from the Military Police of Minas Gerais, Brazil. Due to privacy and legal restrictions, the original data cannot be shared publicly. You can, however, test GRINDA on authorized or synthetic data using the same structure.

---

## 📫 Contact

For questions, collaborations, or partnerships:
	•	Author: Alex S. O. Toledo
	•	Email: [alex.toledo@ibsp.org.br]
	•	Institution: Centro Federal de Educação TEcnológica de Minas Gerais / Instituto Brasileiro de Segurança Pública

---

## 📖 Citation

If you use GRINDA in your research or projects, please cite it as follows: 
Toledo, Alex S. O. "Graph-based Representation for Intelligence and Network Disruption Analysis." Available at: https://github.com/asotoledo/GRINDA.git. 2025.

---

## ⚙️ Installation

1.	Install Anaconda (recommended for managing Python environments and scientific packages)
2.	Clone the repository and install dependencies:

```bash
git clone https://github.com/asotoledo/GRINDA.git
cd grinda

pip install -r requirements.txt


