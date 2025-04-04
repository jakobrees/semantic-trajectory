import numpy as np
import torch
from typing import Union, Tuple, List, Optional

# =============================================================================
# ========================= Dynamic Time Warping Algo =========================
# =============================================================================

def dtw_embedding_similarity(
	sequence1: Union[List, np.ndarray, torch.Tensor],
	sequence2: Union[List, np.ndarray, torch.Tensor],
	band_radius: Optional[int] = None,
	distance_metric: str = "cosine"
) -> Tuple[float, List[Tuple[int, int]]]:
	"""
	Compute DTW similarity between two token embedding trajectories using memory-efficient
	implementation with Sakoe-Chiba band constraint.
	
	Args:
		sequence1: First sequence of embeddings (layers x embedding_dim)
		sequence2: Second sequence of embeddings (layers x embedding_dim)
		band_radius: Sakoe-Chiba band radius (defaults to half of shorter sequence)
		distance_metric: Distance metric to use ("euclidean", "cosine", "manhattan")
	
	Returns:
		Tuple containing:
		- DTW distance (float)
		- Warping path as list of (idx1, idx2) tuples
	"""
	# Convert inputs to numpy if they're torch tensors
	if isinstance(sequence1, torch.Tensor):
		sequence1 = sequence1.detach().cpu().numpy()
	if isinstance(sequence2, torch.Tensor):
		sequence2 = sequence2.detach().cpu().numpy()
	
	# double check format 
	sequence1 = np.array(sequence1)
	sequence2 = np.array(sequence2)
	n = len(sequence1)
	m = len(sequence2)
	
	# Set band radius if not provided (half of shorter sequence)
	if band_radius is None:
		band_radius = min(n, m) >> 1
	
	# Define distance function based on metric parameter
	if distance_metric == "euclidean":
		def dist_fn(x, y):
			# Use numpy's built-in stable norm calculation
			return np.linalg.norm(x - y) / np.sqrt(max(x.shape[0], 1))  # Scale by sqrt of dimensions
	elif distance_metric == "cosine":
		def dist_fn(x, y):
			norm_x = np.linalg.norm(x)
			norm_y = np.linalg.norm(y)
			
			# Handle zero vectors
			if norm_x < 1e-10 or norm_y < 1e-10:
				return 1.0
				
			dot_product = np.dot(x, y)
			similarity = dot_product / (norm_x * norm_y)
			
			# Ensure similarity is in [-1, 1] range
			similarity = max(min(similarity, 1.0), -1.0)
			
			return 1.0 - similarity
	elif distance_metric == "manhattan":
		def dist_fn(x, y):
			return np.sum(np.abs(x - y))
	else:
		raise ValueError(f"Unsupported distance metric: {distance_metric}")
	
	# Initialize only two rows for efficiency of DP
	dp = np.full((2, m + 1), np.inf)
	dp[0, 0] = 0  # Base case
	
	# For reconstructing the warping path
	# Instead of storing the full path, we'll store backtracking pointers
	# Format: (prev_row, prev_col)
	backtrack = np.zeros((n + 1, m + 1, 2), dtype=np.int32)
	
	# Fill the dp array using dynamic programming with band constraint
	for i in range(1, n + 1):
		# Toggle between rows (0 and 1) for current row
		curr = i % 2
		prev = (i - 1) % 2
		
		# Reset the current row to infinity
		dp[curr, :] = np.inf
		
		# Determine band boundaries for column j
		j_start = max(1, i - band_radius)
		j_end = min(m + 1, i + band_radius + 1)
		
		for j in range(j_start, j_end):
			# Calculate distance
			cost = dist_fn(sequence1[i-1], sequence2[j-1])
			
			# Find minimum of three possible previous moves
			min_prev_cost = float('inf')
			min_direction = (-1, -1)  # Default invalid value
			
			# Check diagonal move (i-1, j-1)
			if dp[prev, j-1] < min_prev_cost:
				min_prev_cost = dp[prev, j-1]
				min_direction = (i-1, j-1)
			
			# Check vertical move (i-1, j)
			if dp[prev, j] < min_prev_cost:
				min_prev_cost = dp[prev, j]
				min_direction = (i-1, j)
			
			# Check horizontal move (i, j-1)
			if dp[curr, j-1] < min_prev_cost:
				min_prev_cost = dp[curr, j-1]
				min_direction = (i, j-1)
			
			# Update dp value and backtracking pointer
			dp[curr, j] = cost + min_prev_cost
			backtrack[i, j] = min_direction
	
	# Reconstruct warping path from backtracking pointers
	path = []
	i, j = n, m
	
	while i > 0 and j > 0:
		path.append((i-1, j-1))  # Adjust to 0-indexed
		prev_i, prev_j = backtrack[i, j]
		i, j = prev_i, prev_j
	
	# Add the origin if not already in path
	if path and path[-1] != (0, 0):
		path.append((0, 0))
	
	# Reverse path to get correct order
	path.reverse()
	
	# Return DTW distance and warping path
	return dp[n % 2, m], path


# =============================================================================
# =============================== Example Usage ===============================
# =============================================================================

def compare_token_trajectories(
	token_embeddings1: List[List[torch.Tensor]],
	token_embeddings2: List[List[torch.Tensor]],
	token_idx1: int,
	token_idx2: int,
	distance_metric: str = "euclidean"
) -> float:
	"""
	Compare the layer-wise trajectories of two specific tokens.
	
	Args:
		token_embeddings1: First list of token embeddings [token_idx][layer_idx]
		token_embeddings2: Second list of token embeddings [token_idx][layer_idx]
		token_idx1: Index of token in first embeddings
		token_idx2: Index of token in second embeddings
		distance_metric: Distance metric to use
	
	Returns:
		DTW distance between the token trajectories
	"""
	# Extract embedding trajectories for the specified tokens
	trajectory1 = token_embeddings1[token_idx1]  # List of layer embeddings for token1
	trajectory2 = token_embeddings2[token_idx2]  # List of layer embeddings for token2
	
	# Compute DTW distance (ignore path)
	distance, _ = dtw_embedding_similarity(
		sequence1=trajectory1,
		sequence2=trajectory2,
		distance_metric=distance_metric
	)
	
	return distance

if __name__ == "__main__":
	# Create synthetic embedding trajectories
	
	# Simulate two token trajectories across 12 layers with embedding dim 8
	np.random.seed(42)
	
	# Similar trajectories with some variations
	token1_trajectory = np.random.randn(12, 8)  # 12 layers, 8-dim embeddings
	token2_trajectory = token1_trajectory + 0.3 * np.random.randn(12, 8)  # Similar with noise
	
	# Compute DTW similarity
	distance, path = dtw_embedding_similarity(
		token1_trajectory, 
		token2_trajectory,
		distance_metric="euclidean"
	)
	
	print(f"DTW distance: {distance:.4f}")
	print(f"Path length: {len(path)}")