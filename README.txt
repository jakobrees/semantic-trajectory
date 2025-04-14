Okay, here is a `README.txt` file generated from the provided project description and code files. It aims to be thorough, explaining the concept, implementation, setup, and usage.

```plaintext
================================================================================
 Semantic Trajectory Matching for Information Retrieval
================================================================================

**Version:** 0.1.0 (Inferred from development stage)
**Date:** 2023-10-27 (Placeholder)

--------------------------------------------------------------------------------
## Overview
--------------------------------------------------------------------------------

This project implements and explores a novel Information Retrieval (IR) approach termed "Semantic Trajectory Matching". The core idea moves beyond traditional keyword matching (like TF-IDF, BM25) and static contextual embeddings (like standard BERT/SBERT sentence embeddings) by considering the *evolution* of a word's meaning as it's processed through the layers of a large language model (LLM).

The central hypothesis is that the *path* or *trajectory* an important term's representation takes through the LLM's layers encodes valuable information about its contextual meaning and how that meaning is constructed. Documents where key terms follow similar semantic trajectories to query terms are considered more deeply relevant than those merely sharing keywords or final-layer contextual embeddings.

This approach aims to capture a higher-order similarity related to conceptual development, narrative flow, and semantic refinement, potentially offering advantages over methods like:
*   **Bag-of-Words (BoW):** Ignores order and semantic context.
*   **Static/Final Embeddings:** Capture context but miss the *process* of meaning construction.
*   **Late Interaction (e.g., ColBERT):** Compares final-layer token embeddings, but not their layer-wise evolution.
*   **Cross-Encoders:** Potentially capture deep interactions implicitly but are computationally expensive.

Semantic Trajectory Matching seeks a balance, aiming for deeper semantic understanding while maintaining reasonable efficiency by processing queries and documents independently before comparison.

--------------------------------------------------------------------------------
## Core Concept: Semantic Trajectories
--------------------------------------------------------------------------------

1.  **LLM Choice:** The implementation leverages decoder-only transformer models (e.g., Llama 2). Their causal attention mechanism processes text sequentially (left-to-right), naturally modeling the evolution of meaning as context accumulates.
2.  **Representation Extraction:** For key terms in a query or document, representations are extracted not just from the final layer, but from multiple intermediate layers of the LLM (using `extract_embeddings_util.py`). This creates a sequence of embedding vectors for each token instance.
3.  **Defining a Trajectory:** A "Semantic Trajectory" `T` for a specific instance of a token is the sequence of its embedding vectors across selected layers: `T = [v_1, v_2, ..., v_k]`, where `v_i` is the representation at layer `i` (up to a chosen depth `k`).
4.  **Trajectory Comparison:** The similarity between the trajectories of a query token instance and a document token instance is measured using Dynamic Time Warping (DTW), implemented in `dtw_util.py`. DTW finds the optimal alignment between two sequences, accommodating variations in the "speed" of semantic evolution across layers. Cosine distance is typically used to compare individual layer embeddings within the DTW calculation.
5.  **Document Similarity Aggregation:** To get an overall document relevance score, the system employs a method inspired by ColBERT's MaxSim aggregation but adapted for trajectories (`doc_sim.py`):
    *   **Token Filtering:** Non-informative tokens (stopwords, punctuation) are removed (`token_stopword_util.py`). This may be a mistake due to the way that tokens tends to accumulate meaning in the embedding process (specifically for punctuation, however also based on the location of the specific token, more work here may be required).
    *   **Pre-filtering (Optional):** Static (e.g., first layer) embeddings can be used for a quick cosine similarity check to find the top-k candidate document tokens for each query token, reducing the number of expensive DTW comparisons.
    *   **Instance Matching:** For each unique, filtered query token, its trajectory (or trajectories, if multiple occurrences are considered) is compared against the trajectories of candidate document tokens using DTW.
    *   **MaxSim per Query Token:** The maximum similarity found between any instance of the query token and any instance of a document token is taken as the score for that query token. If token pre-filtering is used, then this provcess is only done for the found top-k tokens.
    *   **Weighted Aggregation:** The final document score is a *weighted average* of these per-query-token maximum similarities. The weights are derived from the inverse frequency of the query tokens in a large corpus (calculated by `token_freq.py`), giving more importance to rarer, potentially more discriminative terms. This weighting scheme uses pre-computed weights loaded from a `.pkl` file.

--------------------------------------------------------------------------------
## Key Features
--------------------------------------------------------------------------------

*   **Semantic Trajectory Extraction:** Extracts layer-wise embeddings for tokens using decoder-only LLMs (`extract_embeddings_util.py`).
*   **Efficient Model Handling:** `ModelManager` class for loading/unloading LLMs and batch processing (`extract_embeddings_util.py`).
*   **Dynamic Time Warping (DTW):** Compares semantic trajectories using DTW with Sakoe-Chiba band constraint for efficiency (`dtw_util.py`). Supports multiple distance metrics (cosine, euclidean, manhattan).
*   **ColBERT-Inspired Weighted Aggregation:** Calculates document similarity using a weighted MaxSim approach based on token trajectory comparisons (`doc_sim.py`).
*   **Token Frequency Weighting:** Emphasizes rarer query terms using pre-computed inverse frequency weights (`doc_sim.py`, `token_freq.py`).
*   **Token Filtering:** Removes stopwords and punctuation, mapping remaining tokens to their original indices (`token_stopword_util.py`).
*   **Parallelism:** Uses `ThreadPoolExecutor` for parallelizing DTW calculations (`doc_sim.py`).
*   **Device Support:** Automatically detects and uses MPS (Apple Silicon), CUDA, or CPU.

--------------------------------------------------------------------------------
## File Structure
--------------------------------------------------------------------------------

*   `doc_sim.py`: Contains the main logic for calculating document similarity using the weighted semantic trajectory matching approach. Includes functions for calculating similarity matrices, MaxSim aggregation, loading token weights, and the primary `document_similarity_colbert_semantic_weighted` function. Includes an example usage block.
*   `dtw_util.py`: Implements the Dynamic Time Warping (DTW) algorithm specifically for comparing sequences of embedding vectors (trajectories). Includes memory optimization and band constraints.
*   `extract_embeddings_util.py`: Provides the `ModelManager` class to handle loading/unloading Hugging Face transformer models (especially causal LLMs) and efficiently extracting token embeddings across specified layers for single texts or batches.
*   `token_stopword_util.py`: Utility functions for tokenizing text, filtering out stopwords (using NLTK) and punctuation, and creating a mapping from unique filtered tokens back to their original indices in the token list.
*   `token_freq.py`: A script to calculate token frequencies across large datasets (e.g., Wikipedia, C4, BookCorpus) using a specified tokenizer (e.g., Llama 2). It generates `.pkl` files containing token counts and derived weights (e.g., log inverse frequency) needed by `doc_sim.py`.
*   `token_frequency_data/` (Directory): Expected location for output files generated by `token_freq.py`, including the crucial `*_weights.pkl` file used for weighting in `doc_sim.py`.

--------------------------------------------------------------------------------
## Setup / Installation
--------------------------------------------------------------------------------

1.  **Python:** Ensure you have Python 3.8+ installed.
2.  **Dependencies:** Install the required libraries:
    ```bash
    pip install torch transformers numpy nltk datasets tqdm
    ```
	alternatively, see the requirements.txt file
3.  **NLTK Data:** Download the NLTK stopwords corpus:
    ```python
    import nltk
    nltk.download('stopwords')
    ```
4.  **LLM Access:** Ensure you have access to the Hugging Face model specified in the scripts (e.g., `meta-llama/Llama-2-7b-hf`). You might need to log in via `huggingface-cli login` if it's a gated model.
5.  **Token Frequency Weights:** This is crucial for the weighted similarity calculation.
    *   You need a token frequency weight file (e.g., `token_frequency_data/llama2_token_freq_weights.pkl`).
    *   This file can be generated by running the `token_freq.py` script. See the "Token Frequency Calculation" section below.
    *   Ensure the `weights_filepath` argument in `document_similarity_colbert_semantic_weighted` (in `doc_sim.py`) points to the correct file.

--------------------------------------------------------------------------------
## Usage
--------------------------------------------------------------------------------

The primary function for calculating similarity between two texts (a query and a document) is `document_similarity_colbert_semantic_weighted` in `doc_sim.py`.

**Basic Workflow:**

1.  **Initialize `ModelManager`:** Create an instance of `ModelManager` with the desired LLM.
2.  **Extract Embeddings & Tokens:** Use the `ModelManager`'s `get_embeddings` method (or `process_batch` for multiple texts) to get the layer-wise embeddings and raw token lists for both the query and the document. Specify the desired `max_layer_depth` and `layer_step`. It is strongly recommended that you pre-process and store document's token-embedding-trajectories as these tend to get extremely large, and are quite expensive to compute.
3.  **Filter Tokens:** Use `get_sorted_filtered_tokens` from `token_stopword_util.py` to get the unique, sorted, filtered tokens and their index mappings for both query and document.
4.  **Calculate Similarity:** Call `document_similarity_colbert_semantic_weighted`, providing the embeddings, sorted filtered tokens, token index mappings for both query and document, and the path to the token weights file.

**Example Snippet (Conceptual - see `doc_sim.py` `if __name__ == "__main__":` for a runnable example):**

```python
from extract_embeddings_util import ModelManager
from token_stopword_util import get_sorted_filtered_tokens
from doc_sim import document_similarity_colbert_semantic_weighted

# --- Configuration ---
MODEL_ID = "meta-llama/Llama-2-7b-hf"
MAX_LAYER_DEPTH = 15 # Example: Use layers 0, layer_step, 2*layer_step, ..., up to 15
LAYER_STEP = 3
WEIGHTS_FILE = "token_frequency_data/llama2_token_freq_weights.pkl"
QUERY_TEXT = "Benefits of swift recovery protocols after traumatic injury"
DOC_TEXT = "This document discusses rapid rehabilitation methods..."

# --- Processing ---
all_embeddings = []
all_sorted_tokens = []
all_token_indices = []

with ModelManager(model_id=MODEL_ID) as manager:
    for text in [QUERY_TEXT, DOC_TEXT]:
        # 1 & 2: Extract Embeddings & Raw Tokens
        embeddings, tokens_raw = manager.get_embeddings(
            input_text=text,
            max_layer_depth=MAX_LAYER_DEPTH,
            layer_step=LAYER_STEP
        )
        # Adjust for potential BOS/EOS differences if necessary (simple example)
        # Actual code might need more robust handling based on tokenizer specifics
        if len(embeddings) != len(tokens_raw):
             min_len = min(len(embeddings), len(tokens_raw))
             embeddings = embeddings[:min_len]
             tokens_raw = tokens_raw[:min_len] # Assuming alignment issue at ends

        # 3: Filter Tokens
        sorted_tokens, token_indices = get_sorted_filtered_tokens(text, manager.tokenizer) # Pass raw text and tokenizer

        all_embeddings.append(embeddings)
        all_sorted_tokens.append(sorted_tokens)
        all_token_indices.append(token_indices)

# 4: Calculate Similarity
similarity_score = document_similarity_colbert_semantic_weighted(
    embeddings1=all_embeddings[0],      # Query embeddings
    sorted_tokens1=all_sorted_tokens[0], # Query filtered tokens
    token_indices1=all_token_indices[0], # Query token indices
    embeddings2=all_embeddings[1],      # Doc embeddings
    sorted_tokens2=all_sorted_tokens[1], # Doc filtered tokens
    token_indices2=all_token_indices[1], # Doc token indices
    weights_filepath=WEIGHTS_FILE,
    # Other parameters like distance_metric, top_k_initial, etc. can be set
)

print(f"Semantic Trajectory Similarity: {similarity_score:.4f}")
```

--------------------------------------------------------------------------------
## Token Frequency Calculation (`token_freq.py`)
--------------------------------------------------------------------------------

The `token_freq.py` script is essential for generating the token weight file used in the similarity calculation. It performs the following:

1.  **Loads Large Datasets:** Iterates through specified datasets (e.g., Wikipedia, C4, BookCorpus) using the Hugging Face `datasets` library in streaming mode to handle large sizes.
2.  **Tokenizes Text:** Uses the specified model's tokenizer (e.g., Llama 2) to tokenize text from the datasets.
3.  **Counts Token Occurrences:** Maintains a count of how many times each token ID appears across the processed corpus.
4.  **Handles Resumption:** Saves progress periodically and can resume counting if interrupted.
5.  **Consolidates Counts:** Merges counts from different datasets into a single file.
6.  **Calculates Weights:** Computes different types of weights based on frequency, such as:
    *   `log_weights`: $\log(\frac{N}{\text{count} + 1})$, similar to IDF.
    *   `reciprocal_weights`: $\frac{1}{\text{frequency} + \epsilon}$.
7.  **Saves Weights:** Stores the counts, frequencies, and calculated weights in a `.pkl` file (e.g., `token_frequency_data/llama2_token_freq_weights.pkl`).

**To Run `token_freq.py`:**

1.  **Configure:** Modify the `DATASET_CONFIGS`, `MODEL_NAME`, `OUTPUT_DIR`, `BATCH_SIZE`, etc., variables at the top of the script as needed. Set `gb_limit` for each dataset (use `-1` to process the entire dataset, `0` to skip).
2.  **Execute:** Run the script from your terminal: `python token_freq.py`
3.  **Wait:** Processing large datasets can take significant time (hours or even days depending on the data size and hardware).
4.  **Output:** The script will create files in the specified `OUTPUT_DIR`, including the final `*_weights.pkl` file required by `doc_sim.py`.

--------------------------------------------------------------------------------
## Potential Future Work / Directions
--------------------------------------------------------------------------------

Based on the initial conceptualization:

*   **General Optimizations:** The dimension of the llama2-7b stream of 4096 result in extremely large document ecodings, which in turn are very computationally expensive, but easily parallelizable given the right hardware. Further optimizatiosn here may be a good idea. 
*   **Composite Similarity Score:** Implement and tune the proposed `Similarity(T_q, T_d) = α·EndpointSim + β·PathSim + γ·InflectionSim` score, exploring methods beyond pure DTW path cost for `PathSim` and `InflectionSim`.
*   **Parameter Tuning:** Optimize hyperparameters like `max_layer_depth`, `layer_step`, `top_k_initial`, `similarity_threshold`, DTW `band_radius`, and potentially the weighting scheme.
*   **Alternative Distance Metrics:** Investigate other sequence comparison metrics beyond DTW (e.g., Frechet distance) or embedding distance metrics.
*   **Evaluation:** Rigorously evaluate the approach on standard IR benchmarks (e.g., MS MARCO, TREC Deep Learning) comparing it against baselines like BM25, SBERT, ColBERT, and Cross-Encoders.