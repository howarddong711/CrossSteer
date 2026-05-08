import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
ASSET_DIR = os.path.join(REPO_ROOT, "assets")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
THIRD_PARTY_DIR = os.path.join(BASE_DIR, "third_party")
TRANSFORMERS_DIR = os.path.join(THIRD_PARTY_DIR, "transformers", "src")
