# Dynamic Token Warping (DTW) Document Similarity

This repository contains a proof-of-concept implementation of a document similarity approach using token embedding trajectories across transformer model layers.

## Overview

The system extracts token embeddings across multiple layers of a language model and compares documents by calculating the similarity between token embedding trajectories using Dynamic Time Warping (DTW).

**Note: This is a proof of concept and has significant room for optimization. The current implementation uses several shortcuts to make computation feasible on consumer hardware.**

## Files

### 1. `extract_embeddings_util.py`
- Contains `ModelManager` class for efficient loading and unloading of transformer models
- Handles extraction of token embeddings across specified layers
- Provides batch processing functionality
- Main functions:
	- `ModelManager.get_embeddings()`: Extract token embeddings across layers
	- `process_batch()`: Process multiple texts with single model loading, can also do multiple documents at once given there is enough hardware support (memory)

### 2. `token_stopword_util.py`
- Handles token filtering and indexing
- Removes stopwords and punctuation from token lists
- Main functions:
	- `get_filtered_tokens()`: Filter tokens based on stopword lists
	- `get_sorted_filtered_tokens()`: Returns sorted unique tokens and their indices

### 3. `dtw_util.py`
- Implements the Dynamic Time Warping algorithm for embedding trajectory comparison
- Main functions:
	- `dtw_embedding_similarity()`: Compute DTW similarity between embedding trajectories
	- `compare_token_trajectories()`: Compare trajectories of two specific tokens

### 4. `doc_sim.py`
- Main script demonstrating document similarity calculation
- Implements the ColBERT-inspired semantic similarity method
- Main functions:
	- `document_similarity_colbert_semantic_avg()`: Calculate document similarity
	- `calculate_similarity_matrix()`: Create pairwise similarity matrix
	- `max_similarity_aggregation()`: Aggregate similarity scores

### 5. `model_download.py`
- Utility script to download the Llama-2-7b model from Hugging Face
- Requires a valid Hugging Face token

## Running the Demo

To run a quick demonstration:
```
python doc_sim.py
```

This will process sample texts, extract their embeddings, and calculate similarity scores between them using the ColBERT-like semantic approach.

## Computational Optimizations

Several optimizations are used to make computation feasible:

1. **Banded DTW**: Uses Sakoe-Chiba band constraint to limit the search space in Dynamic Time Warping
2. **Top-k pre-filtering**: Only performs expensive DTW computations on the top-k most promising token matches based on initial cosine similarity
3. **Similarity threshold cutoff**: Ignores token similarities below a certain threshold (0.75 by default)
4. **Layer sampling**: Allows processing a subset of model layers (using layer_step parameter) to reduce computational load later on. Furthermore there is no point in loading beyond 15 layers as reasearch has shown that the retrievability of the embedding falls off 
5. **Static embedding pre-filtering**: Uses the first layer embeddings as static representations for initial similarity filtering

## Requirements

- PyTorch
- Transformers
- NLTK (for stopwords)
- Access to Hugging Face model hub

(see requirements.txt for full list)

## Future Improvements

- Move Similarity Calculations to GPU
- IDF for token weighting in final similarity weighting 