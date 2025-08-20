# Notebook Documentation: GRINDA_Results.ipynb

This notebook is part of the GRINDA project and is responsible for aggregating, analyzing, and visualizing the results produced by the main GRINDA scripts.

## Features

- Loads processed data and results from previous analysis steps.
- Aggregates and summarizes network metrics and individual statistics.
- Visualizes key findings using tables and plots.
- Allows interactive exploration of results.

## Usage

1. Ensure all required data and result files are present in the `data` and `results` directories.
2. Open the notebook in Jupyter or VS Code.
3. Run the cells sequentially to reproduce the analysis and visualizations.

## Main Variables and Functions

- `DATA_DIR`, `OUTPUT_DIR`: `data` and `results` directories.
- `list_networks()`: Function to list available network files.
- `individuals_df`, `component_info`: DataFrames with individual and network component information.
- Visualization cells: Generate plots and tables for analysis.

## Requirements

- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, networkx, tqdm

## Notes

- Make sure the outputs from previous scripts (`GRINDA_1_embeddingsoutlier.py`, `GRINDA_2_disruption.py`, `GRINDA_3_keyroles.py`) are available.
- The notebook can be adapted to include additional analyses or visualizations.
- See the project README for more details about the workflow and data requirements.

---

This document serves as a reference for the use and understanding of the results aggregation and visualization notebook in the GRINDA project.
