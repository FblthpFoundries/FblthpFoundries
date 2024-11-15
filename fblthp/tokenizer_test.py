from tokenizers import Tokenizer
from train import MagicCardDataset
from torch.utils.data import DataLoader
special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
special_tokens.extend([
    "<tl>","<name>","<mc>","<ot>","<power>","<toughness>","<loyalty>","<ft>", "<nl>",
    "<\\tl>", "<\\name>", "<\\mc>", "<\\ot>", "<\\power>", "<\\toughness>", "<\\loyalty>", "<\\ft>",
    "{W}", "{U}", "{B}", "{R}", "{G}", "{C}", "{X}", "{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}", 
    "{8}", "{9}", "{10}", "{11}", "{12}", "{13}", "{14}", "{15}", "+1/+1"
])
# Load the saved tokenizer
tokenizer = Tokenizer.from_file("wordpiece_tokenizer.json")
max_len = 125
batch_size = 4
dataset = MagicCardDataset(tokenizer, max_len=max_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Test encoding and decoding with special tokens
test_text = "This is a test <name> {W} {U} +1/+1"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)

print("Encoded IDs:", encoded.ids)
print("Decoded text:", decoded)
# Check if each special token has an ID
for token in special_tokens:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        print(f"Token {token} is not in the vocabulary!")
    else:
        print(f"Token: {token}, ID: {token_id}")

print(next(iter(dataloader)))