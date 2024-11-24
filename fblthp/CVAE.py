
from huggingface_hub import HfApi
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoTokenizer
import multiprocessing
from transformers.modeling_utils import PreTrainedModel, Conv1D, prune_conv1d_layer, SequenceSummary
from transformers.modeling_gpt2 import *
from transformers.modeling_bert import gelu
from transformers.configuration_gpt2 import GPT2Config
from transformers.file_utils import add_start_docstrings

def createTokenizer():
    dataset = load_dataset('csv', data_files='corpus.csv', split='train')

    print(dataset[100])
    print(dataset)

    tokenizer_id="gpt2-2024-mtg"

    def batch_iterator(batch_size=10000):
        for i in tqdm(range(0, len(dataset), batch_size)):
            yield dataset[i : i + batch_size]['card']

    # create a tokenizer from existing one to re-use special tokens
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    gpt_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
    gpt_tokenizer.save_pretrained("tokenizer")

    gpt_tokenizer.push_to_hub(tokenizer_id)


class Encoder(GPT2Model):
    def __init__(self, config):
        super(GPT2Model, self).__init__(config)


if __name__ == "__main__":

    user_id = HfApi().whoami()["name"]

    print(f"user id '{user_id}' will be used during the example")

    #createTokenizer()


