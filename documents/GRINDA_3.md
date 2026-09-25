# Script Documentation: GRINDA_3_keyroles.py

This script is part of the GRINDA project and is responsible for identifying key roles and influential individuals within criminal network components.

## Features

- Loads network and individual data.
- Calculates centrality measures and other metrics to identify key nodes.
- Analyzes the roles and influence of individuals in the network.
- Saves results and visualizations for further analysis.

## Usage

1. Ensure the `data` and `result` directories exist and contain the required files.
2. Run the script from the terminal:
   ```bash
   python3 src/GRINDA_3_keyroles.py
   ```
3. Results and figures will be saved in the `result` and `figures` directories.

## Main Variables and Functions

- `DATA_DIR`, `OUTPUT_DIR`: `data` and `result` directories.
- `network_files`: List of network files to process.
- `calculate_centrality`: Function or class to compute centrality measures.
- `key_roles`: Dictionary or DataFrame with identified key individuals and their metrics.

## Requirements

- Python 3.8+
- Libraries: pandas, numpy, networkx, matplotlib, tqdm

## Notes

- The script can be adapted to use different centrality measures or role definitions.
- Results may vary depending on the network structure and chosen metrics.
- See the project README for more details about the data and methodology.

---

This document serves as a reference for the use and understanding of the key roles identification script in the GRINDA project.
