import subprocess
import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

subprocess.run([sys.executable, "src/cleaned_data.py"])
subprocess.run([sys.executable, "src/fields_similarity_scores_generator.py"])
subprocess.run([sys.executable, "src/compatibility_scores.py"])
subprocess.run([sys.executable, "src/matching_algorithms.py"])