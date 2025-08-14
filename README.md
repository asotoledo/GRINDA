# GRINDA - Graph-based Representation for Intelligence and Network Disruption Analysis

> An AI-powered framework to identify critical actors (key roles) in covert criminal networks using GNNs, anomaly detection, and disruption simulation.
>  🚨 The striking name refers to **“grinding down”** (dismantling) and highlights the network's analytical and disruptive focus.

---

## 📌 Overview

Criminal networks, with their complexity and adaptability, pose a significant challenge to traditional investigative methods. **GRINDA** introduces a novel approach that combines Graph Neural Networks (GNNs) — specifically **Graph Attention Networks (GAT)** and **Graph Transformers (GT)** — with advanced outlier detection to identify key actors in these networks using real-world data.

Unlike conventional metrics like centrality measures, our method uncovers hidden actors who may not seem central but are strategically crucial. By incorporating new Deep Learning-based models, GRINDA enhances detection accuracy, while the Neural Network-based classifier allows scalable monitoring of emerging threats.

This research directly aids law enforcement by providing a tool that can **simulate network disruptions** and **predict criminal activities**. Tested on police databases from Minas Gerais, Brazil, this model demonstrates practical relevance and adaptability to multiple crime domains, empowering public security agencies with a proactive, AI-driven framework for resource optimization, strategic intervention, and prevention.

---

## 🎯 Goal

The aim of this work is to develop a **robust, data-driven framework** for identifying **key roles** within criminal networks. By leveraging innovative GNN architectures and advanced anomaly detection techniques, **GRINDA** captures both structural and contextual features of criminal interactions.

The ultimate goal is to enhance law enforcement capabilities in **mapping, disrupting, and monitoring criminal networks** through scalable, intelligent analysis tools — contributing to a safer and more secure society.

---

## 💡 Justification

Traditional methods for analyzing criminal networks often rely on simplistic metrics (degree, betweenness centrality) that may overlook strategically important but less connected individuals. These limitations hinder the ability to fully disrupt complex, decentralized organizations.

With the increasing availability of real-world law enforcement data, there is a pressing need for sophisticated analytical tools capable of capturing the nuanced structure and dynamics of these networks.

GRINDA addresses this gap by integrating **state-of-the-art GNN** and **outlier detection** methods to identify both **central and peripheral** actors critical to criminal activity. Grounded in real police data from Brazil, the approach offers **scalable, operational solutions** directly applicable to intelligence and public security operations.

---

## 🧠 Key Features

• **Graph Transformation**: Converts raw intelligence data into structured graph representations.
• **Deep Node Embeddings**: Learns meaningful embeddings using **GAT** and **GT**.
• **Anomaly Detection**:
	-	Isolation Forest;
	-	Robust Covariance;
	-	SGD One-Class SVM;
	-	Outlier Scores (OS1/OS2);
	-	Cosine Autoencoder Detection Agent;
	-	Deep Support Vector Data Description.
• **Disruption Simulation**: Tests strategic removals to assess structural impact.
• **Scalability**: Modular design for integration with operational intelligence systems.

---

## 🧪 Real-World Application

**GRINDA** was validated on **51 real criminal networks** using police intelligence datasets (REDS and CVO) from the Military Police of Minas Gerais, Brazil.
Due to privacy and legal restrictions, the original datasets are not publicly available. However, the framework can be tested with **authorized or synthetic datasets** following the same structure.

---

## 📫 Contact

For questions, collaborations, or partnerships:
	•	Author: Alex S. O. Toledo
	•	Email: [alex.toledo@ibsp.org.br]
	•	Institution: Centro Federal de Educação Tecnológica de Minas Gerais / Instituto Brasileiro de Segurança Pública

---

## 📖 Citation

If you use GRINDA in your research or projects, please cite it as follows: 
Toledo, Alex S. O. Graph-based Intelligence for Network Disruption and Analysis. Available at: https://github.com/asotoledo/GRINDA.git. 2025.

---

## ⚙️ Installation

1.	Install Anaconda (recommended for managing Python environments and scientific packages)
2.	Clone the repository and install dependencies:

```bash
git clone https://github.com/asotoledo/GRINDA.git
cd grinda

pip install -r requirements.txt


