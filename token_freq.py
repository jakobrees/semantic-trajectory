import os
import torch
import json
import math
import pickle
from tqdm import tqdm
from collections import Counter
from typing import Dict, Optional, Any
from transformers import AutoTokenizer
from datasets import load_dataset
import time

# ====================================================================================
# CONFIGURATION SECTION
# ====================================================================================
# Output directory
OUTPUT_DIR = "token_frequency_data"
# Base name for output files
OUTPUT_BASE_NAME = "llama2_token_freq"
# Model configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
# Device configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
# Dataset configurations
DATASET_CONFIGS = [
	("wikipedia", "20220301.en", "train", -1),            # Process All of Wikipedia 26 Gb (About 2 hours)
	("bookcorpus", None, "train", 0),                   # Process All of BookCorpus (< 6 Gb)
	("c4", "en", "train", 0),                           # Process All of C4 300 Gb (About 15 hours)
	("the_pile", None, "train", 0),                    # Process 0.1 GB of The Pile (800 Gb)
	("wikitext", "wikitext-103-raw-v1", "train", 0),    # Process All of Wikitext (300 Mb)
]
# Simple fixed batch sizes - no dynamic adjustment
BATCH_SIZE = 1024  # For dataset iteration
TOKENIZATION_BATCH_SIZE = 1024  # For tokenization
# Progress saving frequency (in batches)
SAVE_EVERY_N_BATCHES = 10

# ====================================================================================
# HELPER FUNCTIONS
# ====================================================================================
def ensure_output_directory():
	"""Create output directory if it doesn't exist."""
	if not os.path.exists(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)
		print(f"Created output directory: {OUTPUT_DIR}")

def get_progress_filepath(dataset_name: str, config_name: Optional[str]) -> str:
	"""Get the filepath for the progress tracking file."""
	config_str = f"_{config_name}" if config_name else ""
	return os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}_{dataset_name}{config_str}_progress.json")

def get_frequency_filepath(dataset_name: str, config_name: Optional[str]) -> str:
	"""Get the filepath for the dataset's token frequency file."""
	config_str = f"_{config_name}" if config_name else ""
	return os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}_{dataset_name}{config_str}_frequencies.pkl")

def get_consolidated_filepath() -> str:
	"""Get the filepath for the consolidated token frequencies."""
	return os.path.join(OUTPUT_DIR, f"{OUTPUT_BASE_NAME}_consolidated.pkl")

def load_progress(dataset_name: str, config_name: Optional[str]) -> Dict[str, Any]:
	"""Load progress information for a dataset, or create a new one if none exists."""
	filepath = get_progress_filepath(dataset_name, config_name)
	
	if os.path.exists(filepath):
		try:
			with open(filepath, 'r') as f:
				progress = json.load(f)
				print(f"Loaded existing progress for {dataset_name} {config_name or ''}")
				return progress
		except Exception as e:
			print(f"Error loading progress file: {e}")
			print("Starting with new progress tracking.")
	
	# Default progress structure
	return {
		"dataset_name": dataset_name,
		"config_name": config_name,
		"documents_processed": 0,
		"total_tokens_processed": 0,
		"bytes_processed": 0,
		"gb_processed": 0.0,
		"last_updated": time.time(),
		"is_complete": False
	}

def save_progress(progress: Dict[str, Any]) -> None:
	"""Save progress information for a dataset."""
	progress["last_updated"] = time.time()
	filepath = get_progress_filepath(progress["dataset_name"], progress["config_name"])
	
	with open(filepath, 'w') as f:
		json.dump(progress, f, indent=2)

def load_token_frequencies(dataset_name: str, config_name: Optional[str]) -> Counter:
	"""Load existing token frequencies for a dataset, or return empty counter if none exist."""
	filepath = get_frequency_filepath(dataset_name, config_name)
	
	if os.path.exists(filepath):
		try:
			with open(filepath, 'rb') as f:
				return pickle.load(f)
		except Exception as e:
			print(f"Error loading token frequencies: {e}")
			print("Starting with empty token counter.")
	
	return Counter()

def save_token_frequencies(token_counts: Counter, dataset_name: str, config_name: Optional[str]) -> None:
	"""Save token frequencies for a dataset."""
	filepath = get_frequency_filepath(dataset_name, config_name)
	
	with open(filepath, 'wb') as f:
		pickle.dump(token_counts, f)
	
	print(f"Saved token frequencies to {filepath}")

# ====================================================================================
# DATASET PROCESSING FUNCTIONS
# ==================================================================================== 
def process_dataset(dataset_name: str, config_name: Optional[str], split: str, gb_limit: float, tokenizer):
	"""Process a dataset to count token frequencies."""
	
	# Skip processing if gb_limit is 0
	if gb_limit == 0:
		print(f"Skipping {dataset_name} {config_name or ''} (gb_limit = 0)")
		return
	
	print(f"\n{'='*80}")
	if gb_limit > 0:
		print(f"Processing {dataset_name} {config_name or ''}, split: {split}, target: {gb_limit:.2f} GB")
	else:
		print(f"Processing {dataset_name} {config_name or ''}, split: {split}, target: ALL data")
	print(f"{'='*80}")

	# Load progress and token frequencies from previous runs
	progress = load_progress(dataset_name, config_name)
	token_counts = load_token_frequencies(dataset_name, config_name)

	# If dataset is already completely processed, skip it
	if progress["is_complete"]:
		print(f"Dataset {dataset_name} {config_name or ''} is already fully processed.")
		return

	# If we've reached GB limit, mark as complete and skip
	if gb_limit > 0 and progress["gb_processed"] >= gb_limit:
		print(f"Already processed {progress['gb_processed']:.2f} GB, which meets the target of {gb_limit:.2f} GB.")
		progress["is_complete"] = True
		save_progress(progress)
		return

	# Get the last processed index (default to 0 if not found)
	start_index = progress.get("last_index_processed", 0)

	# Load dataset
	try:
		# Standard dataset loading via Hugging Face
		dataset = load_dataset(
			dataset_name, 
			config_name, 
			split=split, 
			streaming=True,
			trust_remote_code=True
		)

		# For resumption, skip documents already processed
		if progress["documents_processed"] > 0:
			print(f"Skipping {progress['documents_processed']} already processed documents...")
			dataset = dataset.skip(progress["documents_processed"])

	except Exception as e:
		print(f"Error loading dataset: {e}")
		return

	# Set processing limits
	if gb_limit > 0:
		gb_remaining = gb_limit - progress["gb_processed"]
		print(f"Need to process {gb_remaining:.2f} more GB to reach target of {gb_limit:.2f} GB")
	else:
		print(f"Processing ALL data in the dataset (currently at {progress['gb_processed']:.2f} GB)")

	# Create batched dataset iterator
	dataset_batched = dataset.batch(BATCH_SIZE)

	# Track metrics
	start_time = time.time()
	batches_processed = 0

	# Create progress bar
	pbar = tqdm(desc=f"Processing {dataset_name} {config_name or ''}", unit='batch')

	# Process the dataset
	try:
		finished_processing_gb = 0
		for batch in dataset_batched:
			# Get text field based on dataset structure
			texts = []
			batch_bytes = 0
			
			# Extract text from various possible fields
			if 'text' in batch:
				texts = batch['text']
			elif 'content' in batch:
				texts = batch['content']
			elif 'sentence' in batch:
				texts = batch['sentence']
			elif 'article' in batch:
				texts = batch['article']
			elif all(isinstance(item, str) for item in batch):
				texts = batch
			
			# Filter out non-string items and empty strings
			texts = [text for text in texts if isinstance(text, str) and text.strip()]
			
			if not texts:
				continue
				
			# Calculate batch size in bytes
			for text in texts:
				batch_bytes += len(text.encode('utf-8'))

			# Tokenize the batch using Hugging Face's efficient batching
			# This will automatically handle the batching internally
			encodings = tokenizer(
				texts,
				add_special_tokens=False,
				padding=False,
				truncation=False,
				return_attention_mask=False,
				return_tensors=None  # Return Python lists
			)

			# Count tokens
			batch_token_count = 0
			for token_ids in encodings["input_ids"]:
				token_counts.update(token_ids)
				batch_token_count += len(token_ids)

			# Update progress
			progress["documents_processed"] += len(texts)
			progress["total_tokens_processed"] += batch_token_count
			progress["bytes_processed"] += batch_bytes
			progress["gb_processed"] = progress["bytes_processed"] / (1024 * 1024 * 1024)

			# Update progress bar
			batches_processed += 1
			pbar.update(1)
			pbar.set_postfix({
				"docs": progress["documents_processed"],
				"GB": f"{progress['gb_processed']:.2f}",
				"tokens": progress["total_tokens_processed"]
			})
			
			# Check if we've reached our GB limit
			if gb_limit > 0 and progress["gb_processed"] >= gb_limit:
				print(f"\nReached GB limit of {gb_limit:.2f} GB")
				progress["is_complete"] = True
				break
			
			# Save progress every Gb
			if progress["gb_processed"] >= finished_processing_gb + 1:
				# update to next requirement 
				finished_processing_gb = progress["gb_processed"]

				save_progress(progress)
				save_token_frequencies(token_counts, dataset_name, config_name)
				
				# Calculate and display processing rate
				elapsed = time.time() - start_time
				if elapsed > 0:
					docs_per_second = progress["documents_processed"] / elapsed
					gb_per_hour = (progress["gb_processed"] * 3600) / elapsed
					print(f"\nProcessing rate: {docs_per_second:.2f} docs/sec, {gb_per_hour:.2f} GB/hour")
			
		# Close progress bar
		pbar.close()
		
		# Mark as complete if we've processed all data or reached GB limit
		if gb_limit > 0 and progress["gb_processed"] >= gb_limit:
			progress["is_complete"] = True
		
		# Save final progress and token frequencies
		save_progress(progress)
		save_token_frequencies(token_counts, dataset_name, config_name)
		
		# Print summary
		print(f"\nFinished processing {dataset_name} {config_name or ''}")
		print(f"  Documents processed: {progress['documents_processed']}")
		print(f"  Total tokens processed: {progress['total_tokens_processed']}")
		print(f"  Total GB processed: {progress['gb_processed']:.3f}")
		print(f"  Unique tokens found: {len(token_counts)}")
		
		elapsed = time.time() - start_time
		if elapsed > 0:
			print(f"  Processing speed: {progress['documents_processed'] / elapsed:.2f} docs/sec")
			print(f"  Data rate: {(progress['bytes_processed'] / elapsed) / (1024*1024):.2f} MB/sec")
	
	except Exception as e:
		# Close progress bar on error
		pbar.close()
		
		print(f"Error during dataset processing: {e}")
		import traceback
		traceback.print_exc()
		
		# Save progress even on error
		print("Saving progress before exiting...")
		save_progress(progress)
		save_token_frequencies(token_counts, dataset_name, config_name)

# ====================================================================================
# TOKEN FREQUENCY ANALYSIS FUNCTIONS
# ====================================================================================
def consolidate_frequencies():
	"""Consolidate token frequencies from all processed datasets."""
	print("\nConsolidating token frequencies from all datasets...")
	
	# Find all frequency files
	frequency_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('_frequencies.pkl') and not f.startswith(f"{OUTPUT_BASE_NAME}_consolidated")]
	
	if not frequency_files:
		print("No frequency files found to consolidate.")
		return
	
	# Load and merge all frequencies
	consolidated_counts = Counter()
	datasets_included = []
	total_tokens_processed = 0
	
	for filename in frequency_files:
		filepath = os.path.join(OUTPUT_DIR, filename)
		try:
			with open(filepath, 'rb') as f:
				dataset_counts = pickle.load(f)
				consolidated_counts.update(dataset_counts)
				
				# Extract dataset info from filename
				base = f"{OUTPUT_BASE_NAME}_"
				dataset_info = filename[len(base):].split('_frequencies.pkl')[0]
				datasets_included.append(dataset_info)
				
				# Count total tokens in this dataset
				dataset_tokens = sum(dataset_counts.values())
				total_tokens_processed += dataset_tokens
				print(f"Added {dataset_info}: {len(dataset_counts)} unique tokens, {dataset_tokens} total tokens")
				
		except Exception as e:
			print(f"Error processing {filename}: {e}")
	
	# Save consolidated frequencies
	consolidated_filepath = get_consolidated_filepath()
	
	# Create metadata about the consolidation
	metadata = {
		"datasets_included": datasets_included,
		"unique_tokens": len(consolidated_counts),
		"total_tokens_processed": total_tokens_processed,
		"date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
	}
	
	# Save both the counts and metadata
	consolidated_data = {
		"metadata": metadata,
		"token_counts": consolidated_counts,
	}
	
	with open(consolidated_filepath, 'wb') as f:
		pickle.dump(consolidated_data, f)
	
	print(f"\nConsolidated {len(datasets_included)} datasets.")
	print(f"Total unique tokens: {len(consolidated_counts)}")
	print(f"Total tokens processed: {total_tokens_processed}")
	print(f"Saved consolidated data to: {consolidated_filepath}")
	
	# Calculate frequency weights
	calculate_weights(consolidated_filepath)

def calculate_weights(consolidated_filepath: str):
	"""Calculate frequency weights from consolidated token counts."""
	print("\nCalculating token frequency weights...")
	
	# Load consolidated data
	with open(consolidated_filepath, 'rb') as f:
		data = pickle.load(f)
	
	token_counts = data["token_counts"]
	metadata = data["metadata"]
	total_tokens = metadata["total_tokens_processed"]
	
	if total_tokens == 0:
		print("No tokens found in consolidated data. Cannot calculate weights.")
		return
	
	# Initialize weight dictionaries
	token_frequencies = {}
	log_weights = {}  # log(N / (count + 1)) style
	reciprocal_weights = {}  # 1 / (freq + ε) style
	
	# Smoothing factor
	epsilon = 1e-9
	
	# Calculate weights for all tokens
	for token_id, count in token_counts.items():
		# Calculate frequency
		frequency = count / total_tokens
		token_frequencies[token_id] = frequency
		
		# Log-based weight (IDF-style)
		log_weight = math.log(total_tokens / (count + 1.0))
		log_weights[token_id] = log_weight
		
		# Reciprocal weight
		recip_weight = 1.0 / (frequency + epsilon)
		reciprocal_weights[token_id] = recip_weight
	
	# Add weight dictionaries to data
	data["token_frequencies"] = token_frequencies
	data["log_weights"] = log_weights
	data["reciprocal_weights"] = reciprocal_weights
	
	# Save enriched data
	weights_filepath = consolidated_filepath.replace("_consolidated.pkl", "_weights.pkl")
	with open(weights_filepath, 'wb') as f:
		pickle.dump(data, f)
	
	print(f"Calculated weights for {len(token_frequencies)} tokens")
	print(f"Saved token frequency weights to: {weights_filepath}")
	
	# Print some example weights
	top_tokens = sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
	bottom_tokens = sorted(token_frequencies.items(), key=lambda x: x[1], reverse=False)[:110]
	
	# Load tokenizer to decode tokens
	try:
		tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
		print("\nExample weights (from most frequent tokens):")
		for token_id, freq in top_tokens:
			token_str = tokenizer.decode([token_id])
			log_w = log_weights[token_id]
			recip_w = reciprocal_weights[token_id]
			print(f"  Token: '{token_str}' (ID: {token_id})")
			print(f"    Frequency: {freq:.6f}")
			print(f"    Log Weight: {log_w:.4f}")
			print(f"    Reciprocal Weight: {recip_w:.4f}")

		print("\n\nExample weights (from least frequent tokens):")
		for token_id, freq in bottom_tokens:
			token_str = tokenizer.decode([token_id])
			log_w = log_weights[token_id]
			recip_w = reciprocal_weights[token_id]
			print(f"  Token: '{token_str}' (ID: {token_id})")
			print(f"    Frequency: {freq:.6f}")
			print(f"    Log Weight: {log_w:.4f}")
			print(f"    Reciprocal Weight: {recip_w:.4f}")
	except Exception as e:
		print(f"Could not decode tokens for display: {e}")

# ====================================================================================
# MAIN EXECUTION
# ====================================================================================
def main():
	"""Main execution function."""
	print(f"Token Frequency Counter for {MODEL_NAME}")
	print(f"Using device: {DEVICE}")
	
	# Ensure output directory exists
	ensure_output_directory()
	
	# Load tokenizer once and reuse
	print(f"Loading tokenizer: {MODEL_NAME}")
	try:
		tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
		vocab_size = tokenizer.vocab_size
		print(f"Loaded tokenizer with vocabulary size: {vocab_size}")
	except Exception as e:
		print(f"Error loading tokenizer: {e}")
		print("Please make sure you have access to the model and it's downloaded.")
		return
	
	# Process each dataset
	for dataset_name, config_name, split, gb_limit in DATASET_CONFIGS:
		process_dataset(dataset_name, config_name, split, gb_limit, tokenizer)
	
	# Consolidate results from all datasets
	consolidate_frequencies()
	
	print("\nToken frequency processing complete!")

if __name__ == "__main__":
	main()