
from huggingface_hub import HfApi

api = HfApi()

def list_files(repo_id):
    print(f"Files in {repo_id}:")
    files = api.list_repo_files(repo_id=repo_id)
    for f in files:
        print(f" - {f}")

list_files("1-800-BAD-CODE/sentence_boundary_detection_multilang")
list_files("1-800-BAD-CODE/punct_cap_seg_47_language")
