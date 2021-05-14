"""
Sometimes there are failed runs, which do not even start.
This script runs on the device and cleans them up.
NOTE: Path to ray_results might be different for on Google Colab.
"""

from pathlib import Path
import shutil
import os

ray_path = Path(f"/home/{os.environ['USER']}/ray_results")

for run_path in ray_path.glob("*/"):
    if len(list(run_path.glob("*"))) == 5:
        shutil.rmtree(run_path)
        print("Here")
