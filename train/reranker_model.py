import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Callable
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
import gc

# Determine the best available device with proper MPS support
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Base class for token weighting strategies
class TokenWeighter(nn.Module):
    """Base class for token weighting strategies"""
    
    def __init__(self):
        super(TokenWeighter, self).__init__()
    
    def forward(self, tokens, token_indices, **kwargs):
        """
        Calculate weights for tokens
        
        Args:
            tokens: List of token strings
            token_indices: Indices of tokens in the sequence
            **kwargs: Additional inputs needed for specific weighting strategies
            
        Returns:
            Tensor of weights for each token
        """
        raise NotImplementedError("Subclasses must implement this method")

    
class VocabLookupWeighter(TokenWeighter):
    """Learns a weight for each token in the vocabulary"""
    
    def __init__(self, vocab_size, init_value=1.0):
        super(VocabLookupWeighter, self).__init__()
        self.token_weights = nn.Parameter(torch.ones(vocab_size) * init_value)
        
    def forward(self, tokens, token_indices, **kwargs):
        """Get weights by vocabulary lookup"""
        # Extract token_ids from kwargs
        token_ids = kwargs.get('token_ids')
        if token_ids is None:
            raise ValueError("VocabLookupWeighter requires token_ids to be provided")
            
        # FIXED: Use index_select to maintain gradient flow
        device = self.token_weights.device
        indices = torch.tensor(token_ids, device=device)
        weights = torch.index_select(self.token_weights, 0, indices)
        return weights


class PositionalWeighter(TokenWeighter):
    """Weights tokens based on their position in the sequence"""
    
    def __init__(self, max_length=512, init_slope=0.01, init_bias=0.5):
        super(PositionalWeighter, self).__init__()
        self.slope = nn.Parameter(torch.tensor(init_slope))
        self.bias = nn.Parameter(torch.tensor(init_bias))
        self.max_length = max_length
        
    def forward(self, tokens, token_indices, **kwargs):
        """Calculate weights based on position"""
        device = self.slope.device
        
        # Normalize positions to [0, 1] - preserving gradient flow
        positions = torch.tensor(token_indices, device=device).float() / self.max_length
        
        # Apply linear transformation: slope * position + bias
        # This creates a new tensor that maintains gradient connection to parameters
        weights = self.slope * positions + self.bias
        
        # Ensure weights are positive
        weights = F.softplus(weights)
        
        return weights


class SurpriseWeighter(TokenWeighter):
    """Weights tokens based on their surprise (-log probability)"""
    
    def __init__(self, init_weight=1.0, init_bias=0.0):
        super(SurpriseWeighter, self).__init__()
        self.weight = nn.Parameter(torch.tensor(init_weight))
        self.bias = nn.Parameter(torch.tensor(init_bias))
        
    def forward(self, tokens, token_indices, **kwargs):
        """
        Calculate weights based on token probabilities
        
        Args:
            tokens: List of token strings
            token_indices: Indices of tokens in the sequence
            token_probs: Token probabilities from the language model
        """
        token_probs = kwargs.get('token_probs')
        if token_probs is None or all(p is None for p in token_probs):
            # Default to uniform weights if no probabilities provided
            return torch.ones(len(tokens), device=self.weight.device)
        
        device = self.weight.device
        
        # Replace None values with a default probability (0.01)
        cleaned_probs = [0.01 if p is None else p for p in token_probs]
        probs = torch.tensor(cleaned_probs, device=device)
        
        # Handle zero probabilities with a small epsilon
        epsilon = 1e-10
        probs = torch.clamp(probs, min=epsilon)
        
        # Calculate surprise: -log(prob)
        surprise = -torch.log(probs)
        
        # Apply weighting: surprise * weight + bias
        # This maintains gradient flow to self.weight and self.bias
        weights = self.weight * surprise + self.bias
        
        # Ensure weights are positive
        weights = F.softplus(weights)
        
        return weights


class CombinedWeighter(TokenWeighter):
    """Combines multiple weighting strategies"""
    
    def __init__(self, weighters, weights=None):
        """
        Initialize combined weighter
        
        Args:
            weighters: List of TokenWeighter instances
            weights: Optional weights for each weighter (will be learned if not provided)
        """
        super(CombinedWeighter, self).__init__()
        self.weighters = nn.ModuleList(weighters)
        
        if weights is None:
            # Create learnable weights for each weighter
            self.strategy_weights = nn.Parameter(torch.ones(len(weighters)))
        else:
            self.strategy_weights = nn.Parameter(torch.tensor(weights))
        
    def forward(self, tokens, token_indices, **kwargs):
        """Combine weights from multiple strategies"""
        device = self.strategy_weights.device
        
        # Calculate weights from each strategy
        all_weights = []
        for weighter in self.weighters:
            weights = weighter(tokens, token_indices, **kwargs)
            all_weights.append(weights)
            
        # Stack weights and apply strategy weighting
        stacked_weights = torch.stack(all_weights)  # [num_strategies, num_tokens]
        strategy_weights = F.softplus(self.strategy_weights).view(-1, 1)  # [num_strategies, 1]
        
        # Combine weights (weighted sum across strategies)
        # This maintains gradient connections to all parameters
        combined = torch.sum(stacked_weights * strategy_weights, dim=0)
        
        return combined


class LlamaReranker(nn.Module):
    """
    Neural reranker using Llama embeddings from a specified layer.
    Uses a flexible late-interaction approach with customizable token weighting.
    Optimized for Apple Silicon MPS.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        layer_idx: int = 20,  # Default to layer 20, can be changed
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        token_weighter: Optional[TokenWeighter] = None,
        similarity_fn: str = "cosine",  # 'cosine', 'dot', 'euclidean'
        weight_normalization: str = "linear"  # 'linear', 'softmax'
    ):
        """
        Initialize the Llama Reranker model.
        
        Args:
            model_name: HuggingFace model identifier
            layer_idx: Which layer's embeddings to extract (0-indexed)
            device: Device to use ("cuda", "mps", "cpu", or None for auto-detection)
            dtype: Data type to use (None for auto-selection based on device)
            max_length: Maximum sequence length to process
            normalize_embeddings: Whether to L2-normalize embeddings
            token_weighter: TokenWeighter instance for weighting tokens
            similarity_fn: Function to use for token similarity calculation
            weight_normalization: Method to normalize token weights ('linear', 'softmax')
        """
        super(LlamaReranker, self).__init__()
        
        # Model configuration
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        
        # Auto-detect device if not specified
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)
        
        # Auto-select dtype if not specified
        if dtype is None:
            if self.device.type == "cpu":
                dtype = torch.float32
            else:
                dtype = torch.float16
        self.dtype = dtype
        
        # Print device info
        print(f"Using device: {self.device} with dtype: {self.dtype}")
        
        # Token weighter
        self.token_weighter = token_weighter
        if self.token_weighter:
            self.token_weighter.to(self.device)
        
        # Similarity and weighting settings
        self.similarity_fn = similarity_fn
        self.weight_normalization = weight_normalization
        
        # Initialize the tokenizer and model
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        
    def load_model(self):
        """Load the tokenizer and model into memory."""
        if self._is_loaded:
            print("Model is already loaded.")
            return
            
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Fix for tokenizers without a pad token
        if self.tokenizer.pad_token is None:
            print("Setting pad_token to eos_token since it was not defined")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model: {self.model_name}")
        
        # Handle MPS-specific loading for HuggingFace models
        if self.device.type == "mps":
            # Load to CPU first and then move to MPS
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",  # Let HF decide initial placement
                output_hidden_states=True
            )
            # Move to MPS manually where needed
        else:
            # Standard loading for CUDA/CPU
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=self.device.type,
                output_hidden_states=True
            )
        
        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        self._is_loaded = True
        print("Model loaded successfully.")
        
    def unload_model(self):
        """Unload the model and tokenizer from memory."""
        if not self._is_loaded:
            print("Model is not currently loaded.")
            return
            
        print("Unloading model...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        # Clear device memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        # MPS doesn't have a direct cache clearing mechanism like CUDA
        # but we can force garbage collection
        gc.collect()
        print("Model unloaded successfully.")
    
    def set_layer_idx(self, layer_idx: int):
        """Change which layer's embeddings to extract."""
        if 0 <= layer_idx <= 31:  # Llama-2 has 32 layers (0-31)
            self.layer_idx = layer_idx
            print(f"Layer index set to {layer_idx}")
        else:
            raise ValueError(f"Invalid layer index: {layer_idx}. Must be between 0 and 31.")
    
    def encode(self, texts: Union[str, List[str]], remove_special_tokens: bool = True) -> Dict:
        """
        Encode texts using the specified layer of Llama.
        
        Args:
            texts: Single text or list of texts to encode
            remove_special_tokens: Whether to remove special tokens from outputs
            
        Returns:
            Dictionary containing:
                - embeddings: Tensor of token embeddings for each text
                - tokens: List of token strings for each text
                - token_ids: List of token IDs for each text
                - token_indices: List of token positions in the sequence
                - token_probs: List of token probabilities (if available)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Handle single text input
        is_single_text = isinstance(texts, str)
        if is_single_text:
            texts = [texts]
            
        # Tokenize inputs
        batch_encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to appropriate device - safe for both CUDA and MPS
        batch_encoding = {k: v.to(self.device) for k, v in batch_encoding.items()}
        
        # Forward pass to get embeddings and logits
        with torch.no_grad():
            outputs = self.model(**batch_encoding, return_dict=True)
            
        # Extract embeddings from the specified layer
        hidden_states = outputs.hidden_states[self.layer_idx]  # [batch_size, seq_len, hidden_dim]
        
        # Get logits for calculating token probabilities if available
        logits = getattr(outputs, 'logits', None)
        if logits is None and hasattr(outputs, 'last_hidden_state'):
            # Use last_hidden_state as a fallback
            logits = outputs.last_hidden_state
        
        # Create results container
        results = []
        
        # Process each text in the batch
        for batch_idx, text in enumerate(texts):
            # Get attention mask for actual tokens (excluding padding)
            attention_mask = batch_encoding["attention_mask"][batch_idx]
            seq_length = attention_mask.sum().item()
            
            # Get token IDs
            token_ids = batch_encoding["input_ids"][batch_idx][:seq_length].tolist()
            
            # Get token strings
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            
            # Get embeddings for this sequence
            text_embeddings = hidden_states[batch_idx, :seq_length].clone()
            
            # Calculate token probabilities (except for the first token)
            token_probs = [None]  # First token has no probability
            
            # Prepare to calculate probabilities if we have logits
            if logits is not None and batch_idx < logits.size(0):
                # MPS-safe way: move to CPU for complex operations that might not be well-supported on MPS
                if self.device.type == "mps" and not torch.is_floating_point(logits):
                    # Handle MPS dtype compatibility issues
                    batch_logits = logits[batch_idx, :seq_length-1].to(torch.device("cpu"), dtype=torch.float32)
                else:
                    batch_logits = logits[batch_idx, :seq_length-1]
                
                # Project logits to vocab size if needed
                if batch_logits.size(-1) != len(self.tokenizer):
                    # Some models return hidden states rather than logits - skip probability calculation
                    token_probs = [None] * seq_length
                else:
                    # Compute softmax on CPU to avoid MPS limitations with certain ops
                    if self.device.type == "mps":
                        batch_probs = F.softmax(batch_logits, dim=-1).cpu()
                    else:
                        batch_probs = F.softmax(batch_logits, dim=-1)
                    
                    # For each position, get probability of the actual next token
                    for pos in range(seq_length - 1):
                        next_token_id = token_ids[pos + 1]
                        prob = batch_probs[pos, next_token_id].item()
                        token_probs.append(prob)
            else:
                # If we can't calculate probabilities, use None for all tokens
                token_probs = [None] * seq_length
            
            # Normalize embeddings if requested
            if self.normalize_embeddings:
                # MPS-compatible normalization
                text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            
            # Filter out special tokens if requested
            if remove_special_tokens:
                # MPS-compatible filtering
                special_tokens = set(self.tokenizer.all_special_tokens)
                keep_mask = [token not in special_tokens for token in tokens]
                
                # Filter embeddings, token_ids, tokens, and probs
                keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
                
                # Empty list check to handle edge cases
                if not keep_indices:
                    # Return empty tensors that won't cause downstream errors
                    filtered_embeddings = text_embeddings.new_zeros((0, text_embeddings.size(1)))
                    filtered_token_ids = []
                    filtered_tokens = []
                    filtered_token_probs = []
                    filtered_token_indices = []
                else:
                    # Use index_select for MPS compatibility instead of boolean indexing
                    select_indices = torch.tensor(keep_indices, device=text_embeddings.device)
                    filtered_embeddings = torch.index_select(text_embeddings, 0, select_indices)
                    filtered_token_ids = [token_ids[i] for i in keep_indices]
                    filtered_tokens = [tokens[i] for i in keep_indices]
                    filtered_token_probs = [token_probs[i] for i in keep_indices]
                    filtered_token_indices = keep_indices
            else:
                filtered_embeddings = text_embeddings
                filtered_token_ids = token_ids
                filtered_tokens = tokens
                filtered_token_probs = token_probs
                filtered_token_indices = list(range(seq_length))
            
            # Add results for this text
            results.append({
                "embeddings": filtered_embeddings,
                "tokens": filtered_tokens,
                "token_ids": filtered_token_ids,
                "token_indices": filtered_token_indices,
                "token_probs": filtered_token_probs
            })
            
        # Return single result for single input
        if is_single_text:
            return results[0]
        
        return results
    
    def calculate_similarity(
        self,
        query_data: Dict,
        doc_data: Dict
    ) -> float:
        """
        Calculate similarity between query and document using weighted MaxSim approach.
        
        Args:
            query_data: Dictionary with query embeddings and tokens
            doc_data: Dictionary with document embeddings and tokens
            
        Returns:
            Similarity score between query and document
        """
        query_embeddings = query_data["embeddings"]
        query_tokens = query_data["tokens"]
        query_token_ids = query_data["token_ids"]
        query_token_indices = query_data["token_indices"]
        query_token_probs = query_data["token_probs"]
        
        doc_embeddings = doc_data["embeddings"]
        doc_tokens = doc_data["tokens"]
        doc_token_ids = doc_data["token_ids"]
        doc_token_indices = doc_data["token_indices"]
        doc_token_probs = doc_data["token_probs"]
        
        # Handle empty embeddings case
        if len(query_embeddings) == 0 or len(doc_embeddings) == 0:
            return 0.0
            
        # MPS compatibility: ensure tensors are on the right device
        if query_embeddings.device != self.device:
            query_embeddings = query_embeddings.to(self.device)
        if doc_embeddings.device != self.device:
            doc_embeddings = doc_embeddings.to(self.device)
        
        # Calculate token-wise similarities - matrix of shape [num_query_tokens, num_doc_tokens]
        if self.similarity_fn == "cosine":
            # Ensure embeddings are normalized if using cosine
            if not self.normalize_embeddings:
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
                doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
            
            # Calculate all similarities at once - MPS compatible
            similarity_matrix = torch.mm(query_embeddings, doc_embeddings.t())
            
        elif self.similarity_fn == "dot":
            similarity_matrix = torch.mm(query_embeddings, doc_embeddings.t())
            
        elif self.similarity_fn == "euclidean":
            # MPS and PyTorch 2.0+ compatible approach for pairwise distances
            try:
                # Try using cdist directly
                similarity_matrix = torch.cdist(query_embeddings, doc_embeddings, p=2)
                # Convert distance to similarity (1 / (1 + distance))
                similarity_matrix = 1.0 / (1.0 + similarity_matrix)
            except RuntimeError:
                # Fallback for MPS if cdist fails
                similarity_matrix = torch.zeros(
                    len(query_embeddings), len(doc_embeddings), 
                    device=self.device
                )
                
                # Compute distances manually (less efficient but more compatible)
                for i in range(len(query_embeddings)):
                    for j in range(len(doc_embeddings)):
                        dist = torch.sqrt(torch.sum((query_embeddings[i] - doc_embeddings[j])**2))
                        similarity_matrix[i, j] = 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"Unsupported similarity function: {self.similarity_fn}")
        
        # For each query token, find the most similar document token
        max_similarities, max_indices = torch.max(similarity_matrix, dim=1)
        
        # Calculate weights for query tokens if weighter is provided
        if self.token_weighter is not None:
            # Ensure weighter is on the correct device
            if next(self.token_weighter.parameters()).device != self.device:
                self.token_weighter.to(self.device)
                
            query_weights = self.token_weighter(
                query_tokens, 
                query_token_indices,
                token_ids=query_token_ids,
                token_probs=query_token_probs
            )
            
            # Get weights for the corresponding document tokens
            best_doc_indices = [doc_token_indices[idx.item()] if idx.item() < len(doc_token_indices) else 0 
                              for idx in max_indices]
            best_doc_tokens = [doc_tokens[idx.item()] if idx.item() < len(doc_tokens) else "" 
                             for idx in max_indices]
            best_doc_token_ids = [doc_token_ids[idx.item()] if idx.item() < len(doc_token_ids) else 0 
                                for idx in max_indices]
            best_doc_token_probs = [doc_token_probs[idx.item()] if idx.item() < len(doc_token_probs) else None 
                                  for idx in max_indices]
            
            doc_weights = self.token_weighter(
                best_doc_tokens,
                best_doc_indices,
                token_ids=best_doc_token_ids,
                token_probs=best_doc_token_probs
            )
            
            # Combine query and document weights (multiply)
            combined_weights = query_weights * doc_weights
            
            # MPS-compatible weight normalization
            if self.weight_normalization == "linear":
                weight_sum = combined_weights.sum()
                if weight_sum > 0:
                    normalized_weights = combined_weights / weight_sum
                else:
                    normalized_weights = torch.ones_like(combined_weights) / len(combined_weights)
                    
            elif self.weight_normalization == "softmax":
                # Make sure inputs to softmax are in a reasonable range to avoid NaNs on MPS
                combined_weights = torch.clamp(combined_weights, -50.0, 50.0)
                normalized_weights = F.softmax(combined_weights, dim=0)
                
            else:
                raise ValueError(f"Unsupported weight normalization: {self.weight_normalization}")
                
            # Calculate weighted sum of similarities
            score = (max_similarities * normalized_weights).sum()
            
        else:
            # No weighting - simple average of max similarities
            score = max_similarities.mean()
            
        return score
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 16,
        return_scores: bool = True
    ) -> Union[List[int], Tuple[List[int], List[float]]]:
        """
        Rerank a list of documents based on similarity to query.
        
        Args:
            query: Query string
            documents: List of document strings to rank
            batch_size: Number of documents to process at once
            return_scores: Whether to return scores along with indices
            
        Returns:
            List of document indices sorted by relevance, and optionally scores
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Encode query
        query_data = self.encode(query)
        
        # Store scores for each document
        scores = []
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            
            # Encode batch of documents
            doc_data_list = self.encode(batch_docs)
            
            # Calculate similarity score for each document
            for j, doc_data in enumerate(doc_data_list):
                score = self.calculate_similarity(query_data, doc_data)
                scores.append(score)
                
        # Get sorted indices (descending by score)
        # Use numpy for sorting to avoid any MPS issues
        scores_np = np.array(scores)
        sorted_indices = np.argsort(-scores_np).tolist()  # Negative for descending order
        
        # Return indices and optionally scores
        if return_scores:
            sorted_scores = [scores[i] for i in sorted_indices]
            return sorted_indices, sorted_scores
        else:
            return sorted_indices
    
    def forward(
        self,
        query_data: Dict,
        doc_data: Dict
    ) -> torch.Tensor:
        """
        Forward pass for training - returns raw similarity score as tensor.
        
        Args:
            query_data: Dictionary with query embeddings and tokens
            doc_data: Dictionary with document embeddings and tokens
            
        Returns:
            Tensor containing similarity score
        """
        return self.calculate_similarity(query_data, doc_data)

    def __enter__(self):
        """Enable context manager usage with 'with' statement."""
        self.load_model()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure model is unloaded when exiting context."""
        self.unload_model()