import numpy as np
from typing import List, Dict, Tuple, Callable, Union, Generator
import torch

# Assume these imports exist and work as before:
from dtw_util import dtw_embedding_similarity
from extract_embeddings_util import ModelManager
from token_stopword_util import get_sorted_filtered_tokens

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

# Keep calculate_similarity_matrix and max_similarity_aggregation as they are
# They likely operate on CPU via numpy / dtw_util
# calculate_similarity_matrix definition from previous code...
def calculate_similarity_matrix(
	trajectories1: List[List[Union[np.ndarray, torch.Tensor]]],
	trajectories2: List[List[Union[np.ndarray, torch.Tensor]]],
	similarity_metric: Callable,
	distance_metric: str = "cosine",
	similarity_threshold: float = 0.75,
) -> np.ndarray:
	"""
	Calculates the pairwise similarity matrix between two lists of embedding trajectories.
	Applies normalization to DTW distance and converts to similarity within this function.
	distance_metric: Distance metric to use ("euclidean", "cosine", "manhattan")
	"""
	# Ensure data is on CPU if similarity_metric expects numpy/cpu
	# This adds overhead if data was on GPU, but necessary if dtw_util is CPU-bound
	trajectories1_cpu = [[layer.cpu().numpy() if isinstance(layer, torch.Tensor) else layer for layer in traj] for traj in trajectories1]
	trajectories2_cpu = [[layer.cpu().numpy() if isinstance(layer, torch.Tensor) else layer for layer in traj] for traj in trajectories2]

	n = len(trajectories1_cpu)
	m = len(trajectories2_cpu)
	similarity_matrix = np.zeros((n, m)) # Result is numpy array on CPU

	for i in range(n):
		# Check if trajectory is empty after potential conversion issues
		if not trajectories1_cpu[i]:
			similarity_matrix[i, :] = 0.0
			continue
		for j in range(m):
			if not trajectories2_cpu[j]:
				similarity_matrix[i, j] = 0.0
				continue
			try:
				raw_distance, _ = similarity_metric(
				trajectories1_cpu[i], trajectories2_cpu[j], distance_metric=distance_metric
				)
				# Simple check for invalid distance (e.g., NaN, Inf)
				if not np.isfinite(raw_distance):
					# print(f"Warning: Non-finite distance ({raw_distance}) encountered for traj pair ({i}, {j}). Setting similarity to 0.")
					similarity = 0.0
				else:
					normalized_distance = raw_distance / (
						len(trajectories1_cpu[i]) + len(trajectories2_cpu[j]) + 1e-9
					)
					similarity = 1.0 / (1.0 + normalized_distance)

					if similarity <= similarity_threshold:
						similarity = 0.0

			except Exception as e:
				# print(f"Warning: Error in similarity_metric for traj pair ({i}, {j}): {e}. Setting similarity to 0.")
				similarity = 0.0 # Default to 0 similarity on error

			similarity_matrix[i, j] = similarity
	return similarity_matrix

# max_similarity_aggregation definition from previous code...
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

# --- New Helper Function for Representative Embeddings ---
def get_representative_embeddings(
	embeddings: List[List[torch.Tensor]], # Expects tensors now
	sorted_tokens: List[str],
	token_indices: Dict[str, List[int]],
	layer_index: int = -1, # Use last layer by default
	aggregation: str = "mean", # 'mean' or 'first' instance
	device: torch.device = DEVICE
) -> Dict[str, torch.Tensor]:
	"""
	Calculates a representative embedding for each unique token type.

	Args:
		embeddings: List[token_idx][layer_idx] -> torch.Tensor (on any device)
		sorted_tokens: List of unique token strings.
		token_indices: Dict mapping token string to list of indices in embeddings.
		layer_index: The layer index to use for representation (-1 for last).
		aggregation: How to aggregate embeddings if a token appears multiple times ('mean', 'first').
		device: The torch device to put the representative embeddings on.

	Returns:
		Dict mapping token string to its representative torch.Tensor on the specified device.
	"""
	representatives = {}
	for token in sorted_tokens:
		indices = token_indices.get(token, [])
		if not indices: continue

		# Collect embeddings for the specified layer for all instances
		instance_embeddings = []
		for idx in indices:
			if idx < len(embeddings) and layer_index < len(embeddings[idx]):
				# Ensure the layer tensor is moved to the target device
				instance_embeddings.append(embeddings[idx][layer_index].to(device))
			# else: print(f"Warning: Index out of bounds for token '{token}' idx={idx} layer={layer_index}")


		if not instance_embeddings: continue # Skip if no valid embeddings found

		# Aggregate instance embeddings
		if aggregation == "mean":
			# Stack tensors and calculate mean along the instance dimension
			stacked_embeddings = torch.stack(instance_embeddings, dim=0)
			rep = torch.mean(stacked_embeddings, dim=0)
		elif aggregation == "first":
			rep = instance_embeddings[0]
		else: # Default to mean
			stacked_embeddings = torch.stack(instance_embeddings, dim=0)
			rep = torch.mean(stacked_embeddings, dim=0)

		representatives[token] = rep # rep is already on the target device
	return representatives

def document_similarity_colbert_semantic_avg(
	# Query (Text 1)
	embeddings1: List[List[Union[np.ndarray, torch.Tensor]]],
	sorted_tokens1: List[str],
	token_indices1: Dict[str, List[int]],
	# Document (Text 2)
	embeddings2: List[List[Union[np.ndarray, torch.Tensor]]],
	sorted_tokens2: List[str],
	token_indices2: Dict[str, List[int]],
	distance_metric: str = "cosine",
	top_k_initial: int = 5,
	similarity_threshold: float = 0.75,
) -> float:
	"""
	Modified version with prefiltering step using static embeddings
	to reduce DTW computation.
	"""
	num_unique_query_tokens = len(sorted_tokens1)
	if num_unique_query_tokens == 0:
		return 0.0  # No query tokens to match
	
	total_max_token_type_similarity = 0.0
	
	# Get static embeddings for efficient pre-filtering
	# We'll use the first layer embeddings as our "static" embeddings
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
	
	# Iterate through each unique token type in the QUERY (text1)
	for query_token_type in sorted_tokens1:
		max_sim_for_current_query_token = 0.0
		query_indices = token_indices1.get(query_token_type, [])
		if not query_indices: continue
		query_trajectories = [embeddings1[idx] for idx in query_indices]
		
		if not sorted_tokens2:  # Handle empty document
			max_sim_for_current_query_token = 0.0
		else:
			# Pre-filtering step: Calculate static embedding similarity
			# and only do DTW for top-k most promising document tokens
			query_static_emb = query_token_static_embeddings[query_token_type]
			
			# Calculate similarity with all document tokens
			token_similarities = []
			for doc_token_type in sorted_tokens2:
				if doc_token_type not in doc_token_static_embeddings:
					continue
					
				doc_static_emb = doc_token_static_embeddings[doc_token_type]
				
				# Calculate static embedding similarity (cosine)
				if isinstance(query_static_emb, torch.Tensor) and isinstance(doc_static_emb, torch.Tensor):
					cos_sim = torch.nn.functional.cosine_similarity(
						query_static_emb.unsqueeze(0), 
						doc_static_emb.unsqueeze(0)
					).item()
				else:  # numpy arrays
					from scipy.spatial.distance import cosine
					cos_sim = 1.0 - cosine(query_static_emb, doc_static_emb)
				
				token_similarities.append((doc_token_type, cos_sim))
			
			# Sort by similarity (descending) and take top_k_initial
			token_similarities.sort(key=lambda x: x[1], reverse=True)
			top_k_tokens = token_similarities[:min(top_k_initial, len(token_similarities))]
			
			# Now only perform DTW on the top-k most promising document tokens
			for doc_token_type, _ in top_k_tokens:
				doc_indices = token_indices2.get(doc_token_type, [])
				if not doc_indices: continue
				doc_trajectories = [embeddings2[idx] for idx in doc_indices]
				
				# DTW calculation
				instance_similarity_matrix = calculate_similarity_matrix(
					query_trajectories, doc_trajectories, 
					similarity_metric=dtw_embedding_similarity,
					distance_metric=distance_metric,
					similarity_threshold = similarity_threshold,
				)
				token_type_similarity_score = max_similarity_aggregation(instance_similarity_matrix)
				
				if token_type_similarity_score > max_sim_for_current_query_token:
					max_sim_for_current_query_token = token_type_similarity_score
		
		# Add to total
		total_max_token_type_similarity += max_sim_for_current_query_token
	
	# Calculate average
	average_max_similarity = total_max_token_type_similarity / num_unique_query_tokens
	return average_max_similarity

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

	print("Calculating ColBERT-Semantic-Avg similarities (Query=Row, Doc=Col)...")
	for i in range(n): # Query i
		print(f"  Processing Query {i+1}/{n}...") # Progress indicator
		for j in range(n): # Document j
			if i == j:
				similarity_matrix_colbert_semantic[i, j] = 1.0 # Similarity with self is max
				continue

			# Calculate similarity using the new function
			similarity_score = document_similarity_colbert_semantic_avg(
				# Query Data (i)
				embeddings1=all_embeddings[i],
				sorted_tokens1=all_sorted_tokens[i],
				token_indices1=all_token_indices[i],
				# Document Data (j)
				embeddings2=all_embeddings[j],
				sorted_tokens2=all_sorted_tokens[j],
				token_indices2=all_token_indices[j],
				top_k_initial=5,
				similarity_threshold=0.75,	# IN PRACTICE WE SHOULD LEARN THESE WITH ML FOR EACH POSSIBLE TOKEN!
			)
			similarity_matrix_colbert_semantic[i, j] = similarity_score

	# Print the ColBERT-Semantic-Avg similarity matrix (Note: Asymmetric!)
	print("\nDocument Similarity Matrix (ColBERT-Semantic-Avg, Query=Row, Doc=Col):")
	print("-" * 75)
	print("      ", end="")
	for j in range(n): print(f"Doc {j+1} ", end="\t")
	print("\n" + "-" * 75)
	for i in range(n):
		print(f"Query {i+1}", end="\t")
		for j in range(n):
			print(f"{similarity_matrix_colbert_semantic[i, j]:.4f}", end="\t")
		print()

	# Print interpretations (highlighting asymmetry)
	print("\nColBERT-Semantic-Avg Similarity Interpretations:")
	print("-" * 75)
	for i in range(n):
		for j in range(n):
			if i == j: continue
			print(f"Query {i+1} -> Document {j+1}: {similarity_matrix_colbert_semantic[i, j]:.4f}")
			# Optional: Print the reverse to show asymmetry
			# print(f"Query {j+1} -> Document {i+1}: {similarity_matrix_colbert_semantic[j, i]:.4f}")
			print(f"  - Query {i+1}:    {texts_to_process[i][:60]}...")
			print(f"  - Document {j+1}: {texts_to_process[j][:60]}...")
			print()