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

	def __enter__(self):
		"""Enable context manager usage with 'with' statement."""
		self.load_model()
		return self
		
	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Ensure model is unloaded when exiting context."""
		self.unload_model()

def process_batch(model_manager: ModelManager, 
				  texts: List[str], 
				  max_layer_depth: Optional[int] = None,
				  layer_step: int = 1,
				  return_numpy: bool = False) -> List[Dict]:
	"""
	Process a batch of texts with a single model loading.
	
	Args:
		model_manager: Initialized ModelManager instance
		texts: List of input texts to process
		max_layer_depth: Maximum layer depth to process
		layer_step: Step size for selecting layers (default: 1).
		return_numpy: Whether to return NumPy arrays
		
	Returns:
		List of dictionaries with results for each text
	"""
	results = []
	
	# Load model once for all texts
	model_manager.load_model()
	
	try:
		for i, text in enumerate(texts):
			print(f"\nProcessing text {i+1}/{len(texts)}")
			embeddings, tokens = model_manager.get_embeddings(
				input_text=text,
				max_layer_depth=max_layer_depth,
				layer_step=layer_step,
				return_numpy=return_numpy
			)
			
			results.append({
				"text": text,
				"tokens": tokens,
				"embeddings": embeddings
			})
	finally:
		# Ensure model is unloaded even if an error occurs
		model_manager.unload_model()
		
	return results


# =============================================================================
# =============================== Example Usage ===============================
# =============================================================================

if __name__ == "__main__":
	"""Example usage of the ModelManager for batch processing."""
	# Example texts
	texts = [
		"The quick brown fox jumps over the lazy dog.",
		"Artificial intelligence is transforming how we interact with technology.",
		"Paris is the capital of France and known for its art and culture.",
	]
	
	# Method 1: Using explicit load/unload
	print("\n=== Method 1: Explicit load/unload ===")
	manager = ModelManager(model_id="meta-llama/Llama-2-7b-hf")
	results = process_batch(
		model_manager=manager,
		texts=texts,
		max_layer_depth=10,
		return_numpy=True
	)
	
	# Method 2: Using context manager
	print("\n=== Method 2: Context manager ===")
	with ModelManager(model_id="meta-llama/Llama-2-7b-hf") as manager:
		for text in texts[:1]:  # Just process the first text as an example
			embeddings, tokens = manager.get_embeddings(
				input_text=text,
				max_layer_depth=10
			)
			print(f"Processed text with {len(tokens)} tokens")
	
	# Display some results
	print("\n=== Results ===")
	for i, result in enumerate(results):
		text = result["text"]
		tokens = result["tokens"]
		embeddings = result["embeddings"]
		
		print(f"\nText {i+1}: '{text[:50]}...' if len(text) > 50 else text")
		print(f"  - Tokens: {len(tokens)}")
		if embeddings:
			print(f"  - Embedding shape: {embeddings[0][0].shape}")
			print(f"  - Layers: {len(embeddings[0])}")
	
	print("\nProcessing complete!")