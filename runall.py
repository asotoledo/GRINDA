import subprocess
import os
import time

start_time = time.time()
scripts = [
    "GRINDA_1_embeddingsoutlier.py",
    "GRINDA_2_disruption.py",
    "GRINDA_3_keyroles.py"
]

src_folder = "src"

for script in scripts:
    script_path = os.path.join(src_folder, script)
    print(f"Running {script_path}...")
    subprocess.run(["python", script_path], check=True)
    print(f"Finished {script_path}...")
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
print(f"Finished all scripts.")