
import os
import shutil
from huggingface_hub import hf_hub_download

def download_and_rename(repo_id, local_dir, file_map):
    print(f"Downloading {repo_id} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)

    # Always download config.yaml
    try:
        print("  Fetching config.yaml...")
        cached_conf = hf_hub_download(repo_id=repo_id, filename="config.yaml")
        shutil.copy2(cached_conf, os.path.join(local_dir, "config.yaml"))
    except Exception as e:
        print(f"  Error downloading config.yaml: {e}")

    for remote_name, local_name in file_map.items():
        try:
            print(f"  Fetching {remote_name} -> {local_name}...")
            cached_path = hf_hub_download(repo_id=repo_id, filename=remote_name)
            dest_path = os.path.join(local_dir, local_name)
            shutil.copy2(cached_path, dest_path)
            print(f"  Saved to {dest_path}")
        except Exception as e:
            print(f"  Error downloading {remote_name}: {e}")

def main():
    # 1. SBD Model
    # Repo: 1-800-BAD-CODE/sentence_boundary_detection_multilang
    # Files: sbd_49lang_bert_small.onnx, spe_mixed_case_64k_49lang.model
    download_and_rename(
        repo_id="1-800-BAD-CODE/sentence_boundary_detection_multilang",
        local_dir=os.path.join("weights", "sbd_multi_lang"),
        file_map={
            "sbd_49lang_bert_small.onnx": "model.onnx",
            "spe_mixed_case_64k_49lang.model": "sp.model"
        }
    )

    # 2. PCS Light Model
    # Repo: 1-800-BAD-CODE/punct_cap_seg_47_language
    # Files: punct_cap_seg_47lang.onnx, spe_unigram_64k_lowercase_47lang.model
    download_and_rename(
        repo_id="1-800-BAD-CODE/punct_cap_seg_47_language",
        local_dir=os.path.join("weights", "punct_cap_seg_47_language"),
        file_map={
            "punct_cap_seg_47lang.onnx": "model.onnx",
            "spe_unigram_64k_lowercase_47lang.model": "sp.model"
        }
    )

    print("\nDownload complete. Checking file sizes...")
    total_size = 0
    over_100mb = False
    for root, dirs, files in os.walk("weights"):
        for f in files:
            fp = os.path.join(root, f)
            size = os.path.getsize(fp)
            total_size += size
            mb_size = size / 1024 / 1024
            print(f"{fp}: {mb_size:.2f} MB")
            if mb_size > 100:
                over_100mb = True

    print(f"\nTotal size: {total_size / 1024 / 1024:.2f} MB")
    if over_100mb:
        print("WARNING: Some files are > 100MB. Git LFS is recommended.")
    else:
        print("All files are < 100MB. Standard Git commit is safe.")

if __name__ == "__main__":
    main()
