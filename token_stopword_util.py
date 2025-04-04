from transformers import AutoTokenizer
from nltk.corpus import stopwords
import string

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation) # Get punctuation characters


# =============================================================================
# ========================= Token Filtering Functions =========================
# =============================================================================

def get_filtered_tokens(text, tokenizer, stop_word_list):
	"""
	Filters tokens via direct stop word check on text substrings
	based on offset mapping. **INDEXING FIXED!**
	
	Args:
		text (str): The input text to tokenize and filter
		tokenizer: The tokenizer object to use for tokenization (must support returning offset mapping)
		stop_word_list (set): A set of stop words to filter out (e.g., NLTK's stopwords)
		
	Returns:
		tuple: Contains two elements:
			- filtered_subword_tokens (list): List of tokens that remain after filtering out stop words and punctuation
			- filtered_indices (list): List of indices corresponding to the positions of the filtered tokens in the 
									  original tokenization output
	"""
	tokenized_output = tokenizer(text, return_offsets_mapping=True, return_attention_mask=False)
	subword_tokens = tokenized_output.tokens()
	offset_mapping = tokenized_output.offset_mapping

	filtered_subword_tokens = []
	filtered_indices = []

	for i, token in enumerate(subword_tokens):
		if i == 0: # skip the <start> token
			continue

		start_offset, end_offset = offset_mapping[i]

		token_substring = text[start_offset:end_offset]
		normalized_substring = token_substring.lower().strip(string.whitespace)
		is_stopword = normalized_substring in stop_word_list
		is_punctuation_only = not token_substring.strip(string.whitespace).isalnum() \
			   					and normalized_substring in punctuation

		if normalized_substring and not is_stopword and not is_punctuation_only:
			filtered_subword_tokens.append(token)
			filtered_indices.append(i)

	return filtered_subword_tokens, filtered_indices

def get_sorted_filtered_tokens(text, tokenizer):
	"""
	Refines the two arrays returned by `get_filtered_tokens`, using the NLTK 
	stopword list for English.
	
	Args:
		text (str): The input text to tokenize and filter
		tokenizer: The tokenizer object to use for tokenization
		
	Returns:
		tuple: Contains two elements:
			- sorted_tokens (list): Alphabetically sorted list of unique filtered tokens
			- token_indices (dict): Dictionary mapping each token to the list of indices 
								   where it appears in the original text
	"""
	# Get filtered tokens and their indices
	filtered_tokens, filtered_indices = get_filtered_tokens(text, tokenizer, stop_words)
	
	# Create a dictionary to map tokens to their indices
	token_to_indices = {}
	for token, idx in zip(filtered_tokens, filtered_indices):
		if token not in token_to_indices:
			token_to_indices[token] = []
		token_to_indices[token].append(idx)
	
	# Get unique tokens and sort them alphabetically
	sorted_tokens = sorted(token_to_indices.keys())
	
	return sorted_tokens, token_to_indices


# =============================================================================
# =============================== Example Usage ===============================
# =============================================================================

if __name__ == "__main__":
	model_id = "meta-llama/Llama-2-7b-hf" # model ID
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	example_text = "The quick brown fox jumps over the lazy dog. Artificial intelligence is amazing!"

	filtered_tokens, filtered_indices = get_filtered_tokens(example_text, tokenizer, stop_words)
	print("Original Tokens:", tokenizer.tokenize(example_text))
	print("Filtered Tokens:", filtered_tokens)
	print("Filtered Token Indices:", filtered_indices)


	# Test the sorted filtered tokens function
	sorted_tokens, token_indices = get_sorted_filtered_tokens(example_text, tokenizer)
	print("\nSorted Unique Tokens:", sorted_tokens)
	print("Token Indices Mapping:")
	for token in sorted_tokens:
		print(f"  {token}: {token_indices[token]}")