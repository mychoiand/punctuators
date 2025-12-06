# Overview

`punctuators` is a project for inference for punctuation and related analytics.

This project is a mostly-undocumented prototype at the moment.

The links to models below contain sufficient documentation for each model.

# Installation

This project can be installed with `pip`:

```bash
$ pip install punctuators
```

# Model Management & Local Usage (New Feature)

We have improved the model loading logic to support **offline usage** and **custom weights**.

## 1. Setup Models
Since the PCS model is large (>200MB), it is **excluded from this Git repository** to avoid LFS issues on public forks. The SBD model (small) is included by default.

To download all necessary models to your local environment, run:

```bash
python download_models.py
```
This will download the models from Hugging Face and save them to the `weights/` directory.

## 2. Model Loading Priority
The library now follows this priority when loading a model (e.g., `"sbd_multi_lang"`):

1.  **Local Directory (Direct Path)**: If you provide a path like `"./my_model"`.
2.  **Local Weights Folder**: Checks if `./weights/{model_name}` exists.
3.  **Hugging Face Hub**: Downloads/Caches from HF (Default behavior if local files are missing).

This allows you to easily switch between local offline models and cloud-hosted models without changing your code.

# Supported Models

This section lists the models currently supported by this package.

## Punctuation, True-Casing, and Sentence Boundary Detection

These models perform punctuation restoration, true-casing (capitalization), and sentence boundary detection (
segmentation).
These analytics together are referred to as PCS (punctuation, capitalization, segmentation).

### 47-language PCS

The following model card describes a base-sized model that can perform PCS on 47 common languages:
https://huggingface.co/1-800-BAD-CODE/punct_cap_seg_47_language

### 47-language PCS (High Accuracy / XLM-Roberta)

The following model card describes a large-sized model that can perform PCS on 47 common languages with higher accuracy:
https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase

## Sentence Boundary Detection

Sentence Boundary Detection (SBD) is the simpler task of accepting punctuated input and segmenting the input into
separate sentences.

### 49-language SBD

The following model card describes a small-sized model that can perform SBD on 49 common languages:
https://huggingface.co/1-800-BAD-CODE/sentence_boundary_detection_multilang


## Usage Examples

This repository includes several scripts to help you test the models easily.

### 1. Interactive Testing (REPL)
Run these scripts to test the models interactively with your own input.

*   **SBD (Sentence Boundary Detection)**: Splits text into sentences.
    ```bash
    python interactive_sbd.py
    ```
*   **PCS (Punctuation, Capitalization, Segmentation)** - Lightweight: Restores punctuation and case, then splits.
    ```bash
    python interactive_pcs.py
    ```
*   **PCS (XLM-Roberta)** - High Accuracy: Heavy model (~1GB) for better accuracy.
    ```bash
    python interactive_xlm.py
    ```

### 2. Static Tests
Run these scripts to see pre-defined examples for English and Korean.

*   `python run_sbd_test.py`: Text compatibility check for SBD.
*   `python run_pcs_test.py`: Text compatibility check for PCS.
*   `python run_xlm_test.py`: Text compatibility check for XLM-Roberta PCS.
