import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(DATA_DIR, 'datasets')
DEFAULT_LINES_PATH = os.path.join(DATASET_DIR, 'lines.txt')
DEFAULT_TOKENIZER_PATH = os.path.join(DATASET_DIR, 'mtg_tokenizer.json')
def get_mtg_tokenizer():
    tokenizer_exists = os.path.exists(DEFAULT_TOKENIZER_PATH)
    if not tokenizer_exists:
        tokenizer = Tokenizer(models.WordPiece())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        special_tokens.extend([
            "<tl>", "<name>", "<mc>", "<ot>", "<power>", "<toughness>", "<loyalty>", "<ft>", "<nl>",
            "<\\tl>", "<\\name>", "<\\mc>", "<\\ot>", "<\\power>", "<\\toughness>", "<\\loyalty>", "<\\ft>",
            "{W}", "{U}", "{B}", "{R}", "{G}", "{C}", "{X}", "{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}", 
            "{8}", "{9}", "{10}", "{11}", "{12}", "{13}", "{14}", "{15}", "+1/+1", "{T}"
        ])
        vocab_size = 20000
        max_len = 125
        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        tokenizer.enable_padding(pad_id=0, pad_token=special_tokens[0], length=max_len)
        tokenizer.enable_truncation(max_length=max_len)
        tokenizer.train([DEFAULT_LINES_PATH], trainer)
        tokenizer.save(DEFAULT_TOKENIZER_PATH)
    else:
        tokenizer = Tokenizer.from_file(DEFAULT_TOKENIZER_PATH)
    return tokenizer