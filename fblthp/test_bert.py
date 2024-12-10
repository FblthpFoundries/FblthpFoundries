from transformers import BertTokenizer, BertModel

# Specify the directory containing the fine-tuned checkpoint
checkpoint_path = "./bert-fine-tuned-mtg-5/checkpoint-29393"

# Load tokenizer and model from the checkpoint
tokenizer = BertTokenizer.from_pretrained(checkpoint_path)
model = BertModel.from_pretrained(checkpoint_path)

# Tokenize input
text = "Hello, how are you?"
tokens = tokenizer(
    text,
    padding=True,
    truncation=True,
    return_tensors="pt",  # Returns PyTorch tensors
)

# Check token IDs
print(tokens["input_ids"])
print(tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])) 