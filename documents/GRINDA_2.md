# Script Documentation: GRINDA_2_disruption.py

This script is part of the GRINDA project and is responsible for simulating and analyzing network disruption strategies in criminal networks.

## Features

- Loads network and individual data.
- Applies disruption strategies to network components (e.g., node removal, edge removal).
- Measures the impact of disruptions on network connectivity and structure.
- Saves results and statistics for further analysis.

## Usage

1. Ensure the `data` and `result` directories exist and contain the required files.
2. Run the script from the terminal:
   ```bash
   python3 src/GRINDA_2_disruption.py
   ```
3. Results will be saved in the `result` directory.

## Main Variables and Functions

- `DATA_DIR`, `OUTPUT_DIR`: `data` and `results` directories.
- `network_files`: List of network files to process.
- `disrupt_network`: Function or class that applies disruption strategies.
- `results`: Dictionary or DataFrame with disruption results and statistics.

## Requirements

- Python 3.8+
- Libraries: pandas, numpy, networkx, matplotlib, tqdm

## Notes

- The script can be adapted to test different disruption strategies.
- Results may vary depending on the chosen strategy and network structure.
- See the project README for more details about the data and methodology.

---

This document serves as a reference for the use and understanding of the network disruption simulation script in the GRINDA project.
