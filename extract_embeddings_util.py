"""
Module for efficient extraction of token embeddings across transformer model layers.

Provides controlled loading/unloading of models for batch processing and
token-centric organization of embeddings for analysis.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Any, Optional, Tuple, Dict
import gc

# Will print out status of operations when true
DEBUG = False

# =============================================================================
# ==================== ModelManager Class & Batch Function ====================
# =============================================================================

class ModelManager:
	"""
	Manager for controlled loading/unloading of transformer models and embedding extraction.
	
	Designed for efficient batch processing of texts by loading the model once
	and extracting embeddings for multiple inputs before unloading.
	"""
	
	def __init__(self, model_id: str = "meta-llama/Llama-2-7b-hf", 
				 device: Optional[str] = None, 
				 dtype: torch.dtype = None) -> None:
		"""
		Initialize the model manager.
		
		Args:
			model_id: HuggingFace model identifier
			device: Device to use ("cuda", "mps", "cpu", or None for auto-detection)
			dtype: Data type to use (None for auto-selection based on device)
		"""
		self.model_id = model_id
		self.tokenizer = None
		self.model = None
		self._is_loaded = False
		
		# Auto-detect device if not specified
		if device is None:
			if torch.cuda.is_available():
				device = "cuda"
			elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
				device = "mps"
			else:
				device = "cpu"
		self.device = device
		
		# Auto-select dtype if not specified
		if dtype is None:
			dtype = torch.float16 if device != "cpu" else torch.float32
		self.dtype = dtype
		
		if DEBUG:
			print(f"Model Manager initialized: {model_id}")
			print(f"  - Device: {device}")
			print(f"  - Data type: {dtype}")

	def load_model(self) -> None:
		"""
		Load the tokenizer and model into memory.
		
		Does nothing if the model is already loaded.
		"""
		if self._is_loaded:
			if DEBUG: print("Model is already loaded.")
			return
			
		if DEBUG: print(f"Loading tokenizer: {self.model_id}")
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
		
		# Fix for tokenizers without a pad token
		if self.tokenizer.pad_token is None:
			if DEBUG: print("Setting pad_token to eos_token since it was not defined")
			self.tokenizer.pad_token = self.tokenizer.eos_token
		
		if DEBUG: print(f"Loading model: {self.model_id}")
		self.model = AutoModelForCausalLM.from_pretrained(
			self.model_id,
			torch_dtype=self.dtype,
			device_map=self.device
		)
		self.model.eval()
		self._is_loaded = True
		if DEBUG: print("Model loaded successfully.")

	def unload_model(self) -> None:
		"""
		Unload the model and tokenizer from memory.
		
		Frees GPU memory if applicable.
		"""
		if not self._is_loaded:
			if DEBUG: print("Model is not currently loaded.")
			return

		if DEBUG: print("Unloading model...")
		del self.model
		del self.tokenizer
		self.model = None
		self.tokenizer = None
		self._is_loaded = False

		# Clear device memory
		if self.device == "cuda":
			torch.cuda.empty_cache()
		gc.collect()
		if DEBUG: print("Model unloaded successfully.")

	def get_embeddings(self, 
					   input_text: str, 
					   max_layer_depth: Optional[int] = None,
					   layer_step: int = 1,
					   return_numpy: bool = False) -> Tuple[List[List[Any]], List[str]]:
		"""
		Extract token-wise embeddings across layers.
		
		Args:
			input_text: Text from which to extract tokens and embedding time series
			max_layer_depth: Maximum layer depth to process (None = all layers)
			layer_step: Step size for selecting layers (e.g., 1 = every layer,
						2 = every second layer). Defaults to 1.
			return_numpy: If True, return NumPy arrays instead of PyTorch tensors
			
		Returns:
			- List of token embeddings [token_idx][layer_idx] -> embedding
			- List of token strings in the same order
			
		Raises:
			RuntimeError: If model is not loaded
			ValueError: If layer_step is not positive.
		"""
		if not self._is_loaded:
			raise RuntimeError("Model not loaded. Call load_model() first.")
		if layer_step <= 0:
			raise ValueError("layer_step must be a positive integer.")
		
		# Tokenize input
		inputs = self.tokenizer(input_text, return_tensors="pt")
		input_ids = inputs["input_ids"]
		inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
		
		# Get token strings
		tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
		if DEBUG: print(f"Tokenized into {len(tokens)} tokens")
		
		# Forward pass with hidden states
		with torch.no_grad():
			outputs = self.model(
				**inputs,
				output_hidden_states=True,
				return_dict=True
			)
		
		# Process hidden states
		all_hidden_states = outputs.hidden_states
		num_layers = len(all_hidden_states)
		sequence_length = input_ids.shape[1]
		
		# Determine the effective maximum layer index to consider
		# Layers are 0-indexed. max_layer_depth=10 means layers 0..10 (11 layers)
		if max_layer_depth is None or max_layer_depth >= num_layers - 1:
			# Process all available layers up to the last one
			effective_stop_layer = num_layers
		else:
			# Process layers up to and including max_layer_depth
			effective_stop_layer = min(max_layer_depth + 1, num_layers)

		# Determine which layers to process using the step
		# range(start, stop, step)
		layers_to_process = range(0, effective_stop_layer, layer_step)

		if DEBUG:
			print(f"Total layers available (incl. embedding): {num_layers}")
			print(f"Requested max_layer_depth: {max_layer_depth}")
			print(f"Effective stop layer (exclusive for range): {effective_stop_layer}")
			print(f"Layer step: {layer_step}")
			print(f"Processing layers at indices: {list(layers_to_process)}")
			print(f"Number of layers to extract per token: {len(list(layers_to_process))}")
		
		# Extract embeddings for each token across layers
		token_embeddings = []
		for token_idx in range(sequence_length):
			# Get embeddings for this token across all requested layers
			token_layers = []
			for layer_idx in layers_to_process:
				vector = all_hidden_states[layer_idx][0, token_idx, :]
				
				# Convert to NumPy or detach PyTorch tensor
				if return_numpy:
					vector = vector.detach().cpu().numpy()
				else:
					vector = vector.detach().cpu()
					
				token_layers.append(vector)
			token_embeddings.append(token_layers)
			
		return token_embeddings, tokens

	def get_embeddings_batch(self, 
						input_texts: List[str],
						text_ids: Optional[List[str]] = None,
						max_layer_depth: Optional[int] = None,
						layer_step: int = 1,
						return_numpy: bool = False) -> List[Dict]:
		"""
		Extract token-wise embeddings across layers for multiple texts in a single batch.
		
		Args:
			input_texts: List of texts from which to extract tokens and embedding time series
			text_ids: Optional list of IDs for each text. If None, indices will be used
			max_layer_depth: Maximum layer depth to process (None = all layers)
			layer_step: Step size for selecting layers (e.g., 1 = every layer,
						2 = every second layer). Defaults to 1.
			return_numpy: If True, return NumPy arrays instead of PyTorch tensors
			
		Returns:
			List of dictionaries, one per text, each containing:
				- text_id: ID of the text
				- text: Original input text
				- tokens: List of token strings
				- embeddings: List of token embeddings [token_idx][layer_idx] -> embedding
			
		Raises:
			RuntimeError: If model is not loaded
			ValueError: If layer_step is not positive or text_ids length doesn't match input_texts
		"""
		if not self._is_loaded:
			raise RuntimeError("Model not loaded. Call load_model() first.")
		if layer_step <= 0:
			raise ValueError("layer_step must be a positive integer.")
		
		# Generate text IDs if not provided
		if text_ids is None:
			text_ids = [str(i) for i in range(len(input_texts))]
		elif len(text_ids) != len(input_texts):
			raise ValueError("Number of text_ids must match number of input_texts")
		
		# Tokenize all inputs
		batch_encoding = self.tokenizer(
			input_texts,
			padding=True,
			truncation=True,
			return_tensors="pt"
		)
		
		# Move batch to the correct device
		batch_encoding = {k: v.to(self.model.device) for k, v in batch_encoding.items()}
		
		# Get the attention mask to identify real tokens vs padding
		attention_mask = batch_encoding["attention_mask"]
		
		if DEBUG:
			print(f"Processing batch of {len(input_texts)} texts")
			print(f"Batch shape: {batch_encoding['input_ids'].shape}")
		
		# Forward pass with hidden states
		with torch.no_grad():
			outputs = self.model(
				**batch_encoding,
				output_hidden_states=True,
				return_dict=True
			)
		
		# Process hidden states
		all_hidden_states = outputs.hidden_states
		num_layers = len(all_hidden_states)
		batch_size, max_seq_length = batch_encoding["input_ids"].shape
		
		# Determine the effective maximum layer index
		if max_layer_depth is None or max_layer_depth >= num_layers - 1:
			effective_stop_layer = num_layers
		else:
			effective_stop_layer = min(max_layer_depth + 1, num_layers)
		
		# Determine which layers to process using the step
		layers_to_process = list(range(0, effective_stop_layer, layer_step))
		
		if DEBUG:
			print(f"Total layers available (incl. embedding): {num_layers}")
			print(f"Processing layers at indices: {layers_to_process}")
			print(f"Number of layers to extract per token: {len(layers_to_process)}")
		
		# Prepare results container
		results = []
		
		# Process each text in the batch
		for batch_idx in range(batch_size):
			text_id = text_ids[batch_idx]
			input_text = input_texts[batch_idx]
			input_ids = batch_encoding["input_ids"][batch_idx]
			
			# Get actual sequence length (excluding padding)
			seq_length = attention_mask[batch_idx].sum().item()
			
			# Get token strings for this text
			tokens = self.tokenizer.convert_ids_to_tokens(input_ids[:seq_length].tolist())
			
			# Extract embeddings for each token across layers
			token_embeddings = []
			for token_idx in range(seq_length):
				# Get embeddings for this token across all requested layers
				token_layers = []
				for layer_idx in layers_to_process:
					vector = all_hidden_states[layer_idx][batch_idx, token_idx, :]
					
					# Convert to NumPy or detach PyTorch tensor
					if return_numpy:
						vector = vector.detach().cpu().numpy()
					else:
						vector = vector.detach().cpu()
						
					token_layers.append(vector)
				token_embeddings.append(token_layers)
			
			# Add results for this text
			results.append({
				"text_id": text_id,
				"text": input_text,
				"tokens": tokens,
				"embeddings": token_embeddings
			})
		
		return results

	def __enter__(self):
		"""Enable context manager usage with 'with' statement."""
		self.load_model()
		return self
		
	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Ensure model is unloaded when exiting context."""
		self.unload_model()

def process_batch(
	model_manager: ModelManager, 
	texts: List[str],
	text_ids: Optional[List[str]] = None, 
	max_layer_depth: Optional[int] = 5,
	layer_step: int = 1,
	return_numpy: bool = False,
	batch_size: int = 4  # Number of texts to process simultaneously
) -> List[Dict]:
	"""
	Process a batch of texts with configurable processing efficiency.
	
	Args:
		model_manager: Initialized ModelManager instance
		texts: List of input texts to process
		text_ids: Optional list of IDs for each text. If None, indices will be used
		max_layer_depth: Maximum layer depth to process
		layer_step: Step size for selecting layers (default: 1)
		return_numpy: Whether to return NumPy arrays
		batch_size: Number of texts to process in a single forward pass
	
	Returns:
		List of dictionaries with results for each text
	"""
	results = []
	
	# Generate text IDs if not provided
	if text_ids is None:
		text_ids = [str(i) for i in range(len(texts))]
	elif len(text_ids) != len(texts):
		raise ValueError("Number of text_ids must match number of texts")
	
	# Efficient processing with batched model loading
	model_manager.load_model()
	try:
		# Process texts in batches
		for batch_start in range(0, len(texts), batch_size):
			batch_end = min(batch_start + batch_size, len(texts))
			batch_texts = texts[batch_start:batch_end]
			batch_ids = text_ids[batch_start:batch_end]
			
			if DEBUG:
				print(f"\nProcessing batch {batch_start//batch_size + 1} "
					  f"(texts {batch_start+1}-{batch_end}/{len(texts)})")
			
			# Get embeddings for the entire batch at once
			batch_results = model_manager.get_embeddings_batch(
				input_texts=batch_texts,
				text_ids=batch_ids,
				max_layer_depth=max_layer_depth,
				layer_step=layer_step,
				return_numpy=return_numpy
			)
			
			# Add batch results to overall results
			results.extend(batch_results)
	finally:
		# Ensure model is unloaded after processing
		model_manager.unload_model()
	
	return results


# =============================================================================
# =============================== Example Usage ===============================
# =============================================================================

if __name__ == "__main__":
	"""Example usage of the ModelManager with new processing options"""
	# Example texts
	texts = [
		"The quick brown fox jumps over the lazy dog.",
		"Artificial intelligence is transforming how we interact with technology.",
		"Paris is the capital of France and known for its art and culture.",
		"Machine learning continues to advance rapidly in various domains.",
		"Quantum computing promises revolutionary computational capabilities."
	]
	
	print("\n=== Efficient Batch Processing ===")
	manager1 = ModelManager(model_id="meta-llama/Llama-2-7b-hf")
	results1 = process_batch(
		model_manager=manager1,
		texts=texts,
		max_layer_depth=10,
		return_numpy=True,
		batch_size=4  # Process 4 texts at a time
	)