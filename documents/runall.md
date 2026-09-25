# Script Documentation: runall.py

This script is part of the GRINDA project and is responsible for automating the execution of the main analysis scripts in sequence.

## Features

- Runs multiple Python scripts in the correct order for the GRINDA workflow.
- Measures and displays the total execution time.
- Provides status messages for each script executed.

## Usage

1. Ensure all required scripts (`GRINDA_1_embeddingsoutlier.py`, `GRINDA_2_disruption.py`, `GRINDA_3_keyroles.py`) are present in the `src` directory.
2. Run the script from the terminal:
   ```bash
   python3 runall.py
   ```
3. The output will display the progress and total execution time.

## Main Variables and Functions

- `scripts`: List of script filenames to execute.
- `src_folder`: Directory containing the scripts.
- `subprocess.run`: Used to execute each script.
- Execution time measurement using `time.time()`.

## Requirements

- Python 3.8+
- Libraries: Standard Python libraries (`os`, `subprocess`, `time`)

## Notes

- Make sure all scripts are executable and have the necessary dependencies installed.
- The script will stop if any of the called scripts fail.
- See the project README for more details about the workflow and data requirements.

---

This document serves as a reference for the use and understanding of the automation script in the GRINDA project.
