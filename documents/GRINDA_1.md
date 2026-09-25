# GRINDA_1_embeddingsoutlier.py

This script is part of the GRINDA project and is responsible for generating graph embeddings and detecting outliers in criminal network components.

## Features

- Loads edge and individual data.
- Builds graphs and identifies connected components.
- Generates embeddings using Graph Attention Networks (GAT) and Graph Transformer (GT).
- Applies outlier detection methods (Isolation Forest, Elliptic Envelope, SVM, OS1, OS2, CoADA, DSVDD).
- Saves results to files for later use.

## Usage

1. Ensure the `data` and `results` directories exist and contain the required files.
2. Run the script from the terminal:
   ```bash
   python src/GRINDA_1_embeddingsoutlier.py
   ```
3. Results will be saved in the `results` directory.

## Main Variables and Functions

- `DATA_DIR`, `OUTPUT_DIR`: `data` and `results` directories.
- `edges_df`: DataFrame with graph edges.
- `component_info`: Information about connected components.
- `GAT`, `GraphTransformer`: Embedding model classes.
- `OutlierDetector`: Outlier detection class.
- `gat_embedding_results`, `gt_embedding_results`: Dictionaries with generated embeddings.
- `outlier_results_gat`, `outlier_results_gt`: Outlier detection results.

## Requirements

- Python 3.8+
- Libraries: pandas, numpy, networkx, torch, torch_geometric, scikit-learn, tqdm

## Notes

- It is recommended to run the script in a GPU-enabled environment for better performance.
- Results may vary due to random initialization of models.
- See the project README for more details about the data structure.

---

This document serves as a reference for the use and understanding of the embedding generation and outlier detection script in the GRINDA project.
