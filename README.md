# Overview

`punctuators` is a project for inference for punctuation and related analytics.

This project is a mostly-undocumented prototype at the moment.

The links to models below contain sufficient documentation for each model.

# Installation

This project can be installed with `pip`:

```bash
$ pip install punctuators
```

# Model Management & Offline Usage

We support **offline usage** via a "Local-First" loading strategy.

*   **Logic**: 1. Custom Path -> 2. Local `weights/` folder -> 3. Hugging Face Hub.
*   **Setup**: Run `python download_models.py` to download models locally.

For detailed instructions and logic diagrams, please refer to:
*   [Model Loading Guide (Korean)](docs/model_loading_ko.md)
*   [Model Loading Guide (English)](docs/model_loading_en.md)

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
