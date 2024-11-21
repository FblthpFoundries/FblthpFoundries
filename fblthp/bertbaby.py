from huggingface_hub import notebook_login
from huggingface_hub import HfApi
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import BertTokenizerFast
import multiprocessing
from transformers import AutoTokenizer
from itertools import chain


if __name__ == "__main__":

    user_id = HfApi().whoami()["name"]

    print(f"user id '{user_id}' will be used during the example")


    dataset = load_dataset('csv', data_files='corpus.csv', split='train')

    # print(dataset[100])
    # print(dataset)

    tokenizer_id="bert-base-uncased-2024-mtg"

    # def batch_iterator(batch_size=10000):
    #     for i in tqdm(range(0, len(dataset), batch_size)):
    #         yield dataset[i : i + batch_size]['card']

    # # create a tokenizer from existing one to re-use special tokens
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
    # bert_tokenizer.save_pretrained("tokenizer")

    # bert_tokenizer.push_to_hub(tokenizer_id)



    tokenizer = BertTokenizerFast.from_pretrained("tokenizer")
    num_proc = multiprocessing.cpu_count() // 2
    print(f"num_proc: {num_proc}")

    def group_texts(examples):
        tokenized_inputs = tokenizer(
        examples["card"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
        )
        return tokenized_inputs 

    tokenized_dataset = dataset.map(group_texts, batched=True, num_proc=num_proc)
    print(tokenized_dataset[100])
