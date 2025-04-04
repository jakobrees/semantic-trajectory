from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

token= "YOUR TOKEN HERE"

if token == "YOUR TOKEN HERE":
    print("Hugging Face Read token required.\n")
    exit()

login(token)

# Download tokenizer and model (with progress bars)
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

print("Downloading model (this will take some time)...")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

print("Model downloaded successfully!")