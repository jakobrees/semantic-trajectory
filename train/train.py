import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import logging
import time
import os
from tqdm import tqdm

# Import your data loader and our reranker
from msmarco_loader import MSMARCOTripleLoader
from reranker_model import LlamaReranker, VocabLookupWeighter, get_device

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('trainer')

def train_reranker(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    layer_idx: int = 3,
    triples_path: str = "./triples.train.small.tsv",
    learning_rate: float = 1e-5,
    batch_size: int = 32,
    steps: int = 1000,
    eval_steps: int = 200,
    output_dir: str = "./model_checkpoints",
    save_steps: int = 250
):
    """Simple training loop for the reranker"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the best available device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer to get vocab size
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Initialize token weighter
    token_weighter = VocabLookupWeighter(vocab_size=vocab_size)
    
    # Initialize reranker model
    reranker = LlamaReranker(
        model_name=model_name,
        layer_idx=layer_idx,
        device=device,  # Explicitly pass device
        token_weighter=token_weighter,
        normalize_embeddings=True,
        weight_normalization="linear"
    )
    
    # Load model
    reranker.load_model()
    
    # Initialize data loader
    data_loader = MSMARCOTripleLoader(
        triples_path=triples_path,
        additional_negatives_per_query=2,
        batch_size=batch_size
    )
    
    # Only train the token weighter parameters
    optimizer = optim.Adam(reranker.token_weighter.parameters(), lr=learning_rate)
    
    # Use MarginRankingLoss
    loss_fn = nn.MarginRankingLoss(margin=0.2)
    
    # Training loop
    reranker.train()
    step = 0
    total_loss = 0
    
    # Start training
    logger.info("Starting training...")
    start_time = time.time()
    
    # Progress tracking
    progress_bar = tqdm(total=steps, desc="Training")
    
    for batch in data_loader.stream_training_data():
        # Group by query
        query_batches = {}
        for query, passage, label in batch:
            if query not in query_batches:
                query_batches[query] = {"pos": [], "neg": []}
            
            if label == 1:
                query_batches[query]["pos"].append(passage)
            else:
                query_batches[query]["neg"].append(passage)
        
        # Process each query
        for query, passages in query_batches.items():
            if not passages["pos"] or not passages["neg"]:
                continue  # Skip if no positive or negative passages
            
            # Get query encoding
            query_data = reranker.encode(query)
            
            # Check if we got valid embeddings (handle edge cases)
            if len(query_data["embeddings"]) == 0:
                logger.warning(f"Empty query embeddings for: {query[:50]}...")
                continue
                
            # Get encodings for positive and negative passages
            pos_data = reranker.encode(passages["pos"][0])  # Take first positive
            if len(pos_data["embeddings"]) == 0:
                logger.warning(f"Empty positive embeddings for: {passages['pos'][0][:50]}...")
                continue
                
            # Process each negative against this positive
            for neg_passage in passages["neg"]:
                neg_data = reranker.encode(neg_passage)
                if len(neg_data["embeddings"]) == 0:
                    logger.warning(f"Empty negative embeddings for: {neg_passage[:50]}...")
                    continue
                    
                # Forward pass
                pos_score = reranker(query_data, pos_data)
                neg_score = reranker(query_data, neg_data)
                
                # Compute loss (positive should rank higher than negative)
                target = torch.tensor(1.0, device=device)
                
                # MPS-specific handling for certain operations
                if device.type == "mps" and (not torch.is_floating_point(pos_score) or not torch.is_floating_point(neg_score)):
                    # Convert to float32 if needed
                    pos_score = pos_score.float()
                    neg_score = neg_score.float()
                
                loss = loss_fn(pos_score, neg_score, target)
                
                # Check for NaN/Inf in loss
                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite loss detected: {loss.item()}. Skipping batch.")
                    continue
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(reranker.token_weighter.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
            
            step += 1
            progress_bar.update(1)
            
            # Logging
            if step % eval_steps == 0:
                avg_loss = total_loss / eval_steps
                elapsed = time.time() - start_time
                logger.info(f"Step {step}/{steps} - Avg Loss: {avg_loss:.4f} - Time: {elapsed:.1f}s")
                total_loss = 0
                start_time = time.time()
                
                # Save checkpoint periodically
                if step % save_steps == 0:
                    save_path = os.path.join(output_dir, f"token_weighter_step_{step}.pt")
                    torch.save(reranker.token_weighter.state_dict(), save_path)
                    logger.info(f"Saved model checkpoint to {save_path}")
            
            # Check if reached max steps
            if step >= steps:
                logger.info(f"Reached maximum steps {steps}")
                break
        
        if step >= steps:
            break
    
    # Close progress bar
    progress_bar.close()
    
    # Save final trained weighter
    final_path = os.path.join(output_dir, "token_weighter_final.pt")
    torch.save(reranker.token_weighter.state_dict(), final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")
    
    # Unload model
    reranker.unload_model()
    
    return reranker.token_weighter

if __name__ == "__main__":
    token_weighter = train_reranker()