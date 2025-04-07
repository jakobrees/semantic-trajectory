import numpy as np
from typing import List, Dict, Callable, Union, Optional
import torch
import pickle
from dtw_util import dtw_embedding_similarity
from  extract_embeddings_util import ModelManager
from concurrent.futures import ThreadPoolExecutor
from token_stopword_util import get_sorted_filtered_tokens
import os

# Check for MPS availability
if torch.backends.mps.is_available():
	DEVICE = torch.device("mps")
elif torch.cuda.is_available():
	DEVICE = torch.device("cuda") # Fallback for NVIDIA
else:
	DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# ================= Document Similarity Functions =============================
# =============================================================================

def calculate_similarity_matrix(
	trajectories1: List[List[Union[np.ndarray, torch.Tensor]]],
	trajectories2: List[List[Union[np.ndarray, torch.Tensor]]],
	similarity_metric: Callable,
	distance_metric: str = "cosine",
	similarity_threshold: float = 0.75,
	max_workers: int = None
) -> np.ndarray:
	"""
	Calculates the pairwise similarity matrix using thread-based parallelism.
	Thread-based approach avoids tokenizer forking issues.
	"""
	# Use number of CPU cores if max_workers not specified
	if max_workers is None:
		max_workers = os.cpu_count() * 2  # Threads can be more numerous than cores
	
	# Ensure data is on CPU
	trajectories1_cpu = [[layer.cpu().numpy() if isinstance(layer, torch.Tensor) else layer for layer in traj] for traj in trajectories1]
	trajectories2_cpu = [[layer.cpu().numpy() if isinstance(layer, torch.Tensor) else layer for layer in traj] for traj in trajectories2]
	
	n = len(trajectories1_cpu)
	m = len(trajectories2_cpu)
	similarity_matrix = np.zeros((n, m))
	
	def compute_similarity(i, j):
		"""Calculate a single similarity value"""
		if not trajectories1_cpu[i] or not trajectories2_cpu[j]:
			return i, j, 0.0
			
		try:
			raw_distance, _ = similarity_metric(
				trajectories1_cpu[i], trajectories2_cpu[j], distance_metric=distance_metric
			)
			
			if not np.isfinite(raw_distance):
				return i, j, 0.0
				
			normalized_distance = raw_distance / (
				len(trajectories1_cpu[i]) + len(trajectories2_cpu[j]) + 1e-9
			)
			similarity = 1.0 / (1.0 + normalized_distance)
			
			if similarity <= similarity_threshold:
				return i, j, 0.0
			else:
				return i, j, similarity
				
		except Exception as e:
			return i, j, 0.0
	
	# Process with ThreadPoolExecutor
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		# Submit all tasks
		futures = []
		for i in range(n):
			for j in range(m):
				futures.append(executor.submit(compute_similarity, i, j))
		
		# Process results as they complete
		for future in futures:
			i, j, similarity = future.result()
			similarity_matrix[i, j] = similarity

	return similarity_matrix

def max_similarity_aggregation(similarity_matrix: np.ndarray) -> float:
	"""
	Performs MaxSim aggregation on a similarity matrix.
	Note that we expect the similarities to already be in [0,1] with 1
	indicating a more similar match. This is similar to a type of crude
	clustering approach: actual clusternig with learned thresholds might
	be ideal.
	Args:
		similarity_matrix: The token similarity matrix.
	Returns:
		float: MaxSim score (average of maximum similarities for each row).
	"""
	if not isinstance(similarity_matrix, np.ndarray):
		 # print(f"Warning: max_similarity_aggregation received non-numpy input type {type(similarity_matrix)}. Converting.")
		try:
			similarity_matrix = np.array(similarity_matrix)
		except Exception as e:
			# print(f"Error converting input to numpy array in max_similarity_aggregation: {e}")
			return 0.0

	if similarity_matrix.size == 0:  # Handle empty matrix case
		return 0.0

	# Check dimensions properly for numpy arrays
	if similarity_matrix.ndim == 0: # Scalar case
		return float(similarity_matrix)
	if similarity_matrix.ndim == 1: # Vector case
		return np.mean(similarity_matrix) if similarity_matrix.size > 0 else 0.0

	# 2D Matrix case
	rows, cols = similarity_matrix.shape
	if rows == 0 or cols == 0:
		return 0.0

	try:
		max_similarities = np.max(similarity_matrix, axis=1) # max over rows (queries)
		return np.mean(max_similarities)
	except Exception as e:
		# print(f"Error during max/mean calculation in max_similarity_aggregation: {e}")
		return 0.0

# Global variables to store token weights after first load
_TOKEN_WEIGHTS = {}
_WEIGHTS_LOADED = False
_WEIGHTS_FILEPATH = None
_WEIGHT_TYPE = None

def load_token_weights(weights_filepath: str, weight_type: str = "log_weights") -> Dict:
	"""
	Load token weights from a pickle file.
	
	Args:
		weights_filepath: Path to the pickle file containing token weights
		weight_type: Type of weight to use ('log_weights' or 'reciprocal_weights')
		
	Returns:
		Dict containing token weights
	"""
	global _TOKEN_WEIGHTS, _WEIGHTS_LOADED, _WEIGHTS_FILEPATH, _WEIGHT_TYPE
	
	# If weights are already loaded from the same file and of the same type, reuse them
	if _WEIGHTS_LOADED and _WEIGHTS_FILEPATH == weights_filepath and _WEIGHT_TYPE == weight_type:
		return _TOKEN_WEIGHTS
	
	# Otherwise, load them from file
	try:
		with open(weights_filepath, 'rb') as f:
			weight_data = pickle.load(f)
			_TOKEN_WEIGHTS = weight_data.get(weight_type, {})
			_WEIGHTS_LOADED = True
			_WEIGHTS_FILEPATH = weights_filepath
			_WEIGHT_TYPE = weight_type
			# print(f"Loaded {len(_TOKEN_WEIGHTS)} token weights from {weights_filepath}")
			return _TOKEN_WEIGHTS
	except Exception as e:
		print(f"Error loading token weights: {e}")
		_WEIGHTS_LOADED = False
		_TOKEN_WEIGHTS = {}
		return {}

def document_similarity_colbert_semantic_weighted(
	# Query (Text 1)
	embeddings1: List[List[Union[np.ndarray, torch.Tensor]]],
	sorted_tokens1: List[str],
	token_indices1: Dict[str, List[int]],
	# Document (Text 2)
	embeddings2: List[List[Union[np.ndarray, torch.Tensor]]],
	sorted_tokens2: List[str],
	token_indices2: Dict[str, List[int]],
	# Configuration
	distance_metric: str = "cosine",
	top_k_initial: int = 5,
	similarity_threshold: float = 0.75,
	weights_filepath: Optional[str] = None,
	weight_type: str = "log_weights",  # 'log_weights' or 'reciprocal_weights'
	fallback_weight: float = 21.5481  # Default weight for tokens not in dictionary
) -> float:
	"""
	Enhanced version of document similarity that uses token frequency weights
	to emphasize rarer tokens in the similarity calculation.
	
	Args:
		embeddings1, sorted_tokens1, token_indices1: Query text representations
		embeddings2, sorted_tokens2, token_indices2: Document text representations
		distance_metric: Metric to use for embedding distance calculation
		top_k_initial: Number of top document tokens to consider for DTW
		similarity_threshold: Minimum similarity threshold
		weights_filepath: Path to the pickle file containing token weights
		weight_type: Type of weight to use ('log_weights' or 'reciprocal_weights')
		fallback_weight: Weight to use for tokens not found in the dictionary
		
	Returns:
		float: Weighted similarity score between query and document
	"""
	num_unique_query_tokens = len(sorted_tokens1)
	if num_unique_query_tokens == 0:
		return 0.0  # No query tokens to match
	
	# Load token weights if filepath is provided (will use cached version if already loaded)
	token_weights = {}
	if weights_filepath:
		token_weights = load_token_weights(weights_filepath, weight_type)
	
	# Extract static embeddings for pre-filtering
	query_token_static_embeddings = {}
	doc_token_static_embeddings = {}
	
	# Extract static embeddings for query tokens
	for query_token_type in sorted_tokens1:
		query_indices = token_indices1.get(query_token_type, [])
		if not query_indices: continue
		# Use the first token instance and first layer as static representation
		query_token_static_embeddings[query_token_type] = embeddings1[query_indices[0]][0]
	
	# Extract static embeddings for document tokens
	for doc_token_type in sorted_tokens2:
		doc_indices = token_indices2.get(doc_token_type, [])
		if not doc_indices: continue
		# Use the first token instance and first layer as static representation
		doc_token_static_embeddings[doc_token_type] = embeddings2[doc_indices[0]][0]
	
	# Store token similarities and their weights
	token_similarities = []
	total_weight = 0.0
	weighted_sum = 0.0
	
	# Iterate through each unique token type in the QUERY (text1)
	for query_token_type in sorted_tokens1:
		query_indices = token_indices1.get(query_token_type, [])
		if not query_indices: continue
		
		# Get weight for this token (use fallback_weight if not found)
		token_id = int(query_token_type) if query_token_type.isdigit() else query_token_type
		token_weight = token_weights.get(token_id, fallback_weight)
		
		query_trajectories = [embeddings1[idx] for idx in query_indices]
		max_sim_for_current_query_token = 0.0
		
		if sorted_tokens2:  # If document has tokens
			# Pre-filtering step using static embeddings
			query_static_emb = query_token_static_embeddings[query_token_type]
			
			# Handle both PyTorch and NumPy versions
			if isinstance(query_static_emb, torch.Tensor):
				# Move to GPU if available
				device = torch.device("mps")  # Use Metal Performance Shaders on M3 Mac
				query_emb = query_static_emb.to(device).unsqueeze(0)
				
				# Get all document embeddings at once
				valid_tokens = [t for t in sorted_tokens2 if t in doc_token_static_embeddings]
				if not valid_tokens:
					top_k_tokens = []
				else:
					# Stack all document embeddings into a single tensor
					doc_embs = torch.stack([doc_token_static_embeddings[t].to(device) for t in valid_tokens])
					
					# Calculate all similarities at once
					similarities = torch.nn.functional.cosine_similarity(query_emb, doc_embs)
					
					# Get top-k indices and similarities
					if len(similarities) <= top_k_initial:
						top_indices = torch.argsort(similarities, descending=True)
					else:
						top_indices = torch.topk(similarities, k=top_k_initial).indices
						
					# Convert back to CPU for further processing
					top_similarities = similarities[top_indices].cpu().tolist()
					top_indices = top_indices.cpu().tolist()
					
					# Create result list
					top_k_tokens = [(valid_tokens[idx], sim) for idx, sim in zip(top_indices, top_similarities)]
			else:
				# NumPy version (can also be optimized)
				query_emb = np.array(query_static_emb).reshape(1, -1)
				
				valid_tokens = [t for t in sorted_tokens2 if t in doc_token_static_embeddings]
				if not valid_tokens:
					top_k_tokens = []
				else:
					# Stack all document embeddings
					doc_embs = np.stack([doc_token_static_embeddings[t] for t in valid_tokens])
					
					# Calculate all similarities at once
					# (1 - distance) is equivalent to cosine similarity
					# Efficient dot product calculation for normalized vectors
					similarities = np.dot(query_emb, doc_embs.T)[0]
					
					# Get top-k indices
					top_indices = np.argsort(similarities)[::-1][:top_k_initial]
					
					# Create result list
					top_k_tokens = [(valid_tokens[idx], similarities[idx]) for idx in top_indices]
			
			# Perform DTW on the top-k most promising document tokens
			for doc_token_type, _ in top_k_tokens:
				doc_indices = token_indices2.get(doc_token_type, [])
				if not doc_indices: continue
				doc_trajectories = [embeddings2[idx] for idx in doc_indices]
				
				# DTW calculation
				instance_similarity_matrix = calculate_similarity_matrix(
					query_trajectories, doc_trajectories, 
					similarity_metric=dtw_embedding_similarity,
					distance_metric=distance_metric,
					similarity_threshold=similarity_threshold,
				)
				token_type_similarity_score = max_similarity_aggregation(instance_similarity_matrix)
				
				if token_type_similarity_score > max_sim_for_current_query_token:
					max_sim_for_current_query_token = token_type_similarity_score
		
		# Store token similarity and its weight
		token_similarities.append((query_token_type, max_sim_for_current_query_token, token_weight))
		weighted_sum += max_sim_for_current_query_token * token_weight
		total_weight += token_weight
	
	# Calculate weighted average
	if total_weight > 0:
		weighted_avg_similarity = weighted_sum / total_weight
	else:
		weighted_avg_similarity = 0.0
	
	return weighted_avg_similarity


# =============================================================================
# =============================== Example Usage ===============================
# =============================================================================

if __name__ == "__main__":
	texts_to_process = [
		"The quick brown fox jumps over the lazy dog.",
		"The quick brown dog jumps over the lazy fox.",
		"The fast dog is running and jumping on the track",
		"This is a completely unrelated text about artificial intelligence.",
		"Paris is the capital of France and known for its art and culture.",
		"Technology is rapidly changing the world around us, but it's overhyped.",

		# This is query
		"what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft.",

		# This is document 486 should be similar to query
		"similarity laws for aerothermoelastic testing. the similarity laws for aerothermoelastic testing are presentedin the range. these are obtained bymaking nondimensional the appropriate governing equations ofthe individual external aerodynamic flow, heat conduction tothe interior, and stress-deflection problems which make up thecombined aerothermoelastic problem. for the general aerothermoelastic model, where the model isplaced in a high-stagnation-temperature wind tunnel, similitudeis shown to be very difficult to achieve for a scale ratio otherthan unity. the primary conflict occurs between thefree-stream mach number reynolds number aeroelasticparameter heat conduction parameter andthermal expansion parameter. means of dealing with this basic conflict are presented. theseinclude (1) looking at more specialized situations, such as thebehavior of wing structures and of thin solid plate lifting surfaces,and panel flutter, where the aerothermoelastic similarityparameters assume less restrictive forms, (2) the use of /incompleteaerothermoelastic/ testing in which the pressure and/or heatingrates are estimated in advance and applied artificially to themodel, and (3) the use of /restricted purpose/ modelsinvestigating separately one or another facet of the completeaerothermoelastic problem. some numerical examples of modeling for the generalaerothermoelastic case as well as for the specialized situationsmentioned in (1) above are given. finally, extension of the aerothermoelastic similarity laws tohigher speeds and temperatures is discussed .",

		# This is document 29: should be less similar to query than above
		"a simple model study of transient temperature and thermal stress distribution due to aerodynamic heating. the present work is concerned with the determination of transient temperatures and thermal stresses in simple models intended to simulate parts or the whole of an aircraft structure of the built- up variety subjected to aerodynamic heating. the first case considered is that of convective heat transfer into one side of a flat plate, representing a thick skin, and the effect of the resulting temperature distribution in inducing thermal stresses associated with bending restraint at the plate edges. numerical results are presented for the transient temperature differentials in the plate when the environment temperature first increases linearly with time and then remains constant, the period of linear increase representing the time of acceleration of the aircraft. corresponding thermal stress information is presented. the second case is that of the wide-flanged i-beam with convective heat transfer into the outer faces of the flanges. numerical results are presented for transient temperature differentials for a wide range of values of the applicable parameters and for an environment temperature variation as described above. corresponding thermal stresses in a beam of infinite length are determined. a theoretical analysis of the stress distribution in a beam of finite length is carried out and numerical results obtained for one case. an experimental investigation of temperatures and stresses in such a beam is described, and results are presented which indicate good agreement with corresponding theoretical results.",
	]
	
	# Initialize matrices for storing all embeddings and tokens
	all_embeddings = []
	all_sorted_tokens = []
	all_token_indices = []
	
	# --- Embedding Extraction ---
	print("Extracting embeddings...")
	try:
		with ModelManager(model_id="meta-llama/Llama-2-7b-hf") as manager:
			for idx, text in enumerate(texts_to_process):
				print(f"  Processing text {idx+1}/{len(texts_to_process)}...")
				# Get embeddings (List[token][layer]) and raw tokens
				embeddings, tokens_raw = manager.get_embeddings(
					input_text=text,
					max_layer_depth=15, # Using 15 layers
					layer_step=3
				)

				# Ensure embeddings and tokens align (handle BOS/EOS differences if any)
				if len(embeddings) != len(tokens_raw):
					print(f"Warning: Mismatch len embeddings ({len(embeddings)}) vs tokens ({len(tokens_raw)}) for doc {idx+1}. Adjusting...")
					# Simple fix: truncate longer list (adjust logic as needed)
					min_len = min(len(embeddings), len(tokens_raw))
					embeddings = embeddings[:min_len]
					tokens_raw = tokens_raw[:min_len]


				# Get filtered unique tokens and their indices in the *original* token list
				# Pass the raw tokens corresponding to the embeddings
				sorted_tokens, token_indices = get_sorted_filtered_tokens(text, manager.tokenizer)

				# Store data
				all_embeddings.append(embeddings)
				all_sorted_tokens.append(sorted_tokens)
				all_token_indices.append(token_indices)

	except NameError:
		print("ERROR: ModelManager or get_sorted_filtered_tokens not defined.")
		print("Please ensure necessary imports and definitions are available.")
		exit()

	print("Embedding extraction complete.")

	# --- Similarity Calculation ---
	n = len(texts_to_process)
	similarity_matrix_colbert_semantic = np.zeros((n, n))

	print("Calculating ColBERT-like-Semantic-Avg similarities (Query=Row, Doc=Col)...")
	for i in range(n): # Query i
		print(f"  Processing Query {i+1}/{n}...") # Progress indicator
		for j in range(n): # Document j
			if i == j:
				similarity_matrix_colbert_semantic[i, j] = 1.0 # Similarity with self is max
				continue

			# Calculate similarity using the new function
			similarity_score = document_similarity_colbert_semantic_weighted(
				# Query Data (i)
				embeddings1=all_embeddings[i],
				sorted_tokens1=all_sorted_tokens[i],
				token_indices1=all_token_indices[i],
				# Document Data (j)
				embeddings2=all_embeddings[j],
				sorted_tokens2=all_sorted_tokens[j],
				token_indices2=all_token_indices[j],
				top_k_initial=5,
				similarity_threshold=0.75,
				weights_filepath="token_frequency_data/llama2_token_freq_weights.pkl",
				weight_type="log_weights",
				fallback_weight=21.5481  # Default weight for tokens not found in dictionary
			)
			similarity_matrix_colbert_semantic[i, j] = similarity_score

	# Print the ColBERT-like-Semantic-Avg similarity matrix (Note: Asymmetric!)
	print("\nDocument Similarity Matrix (ColBERT-like-Semantic-Avg, Query=Row, Doc=Col):")
	print("-" * 75)
	print("      ", end="")
	for j in range(n): print(f"Doc {j+1} ", end="\t")
	print("\n" + "-" * 75)
	for i in range(n):
		print(f"Query {i+1}", end="\t")
		for j in range(n):
			print(f"{similarity_matrix_colbert_semantic[i, j]:.4f}", end="\t")
		print()