import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from csv2tokens import getCorpus
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
device = 'cuda'
block_size = 256
num_epochs = 20

#https://huggingface.co/docs/transformers/tasks/language_modeling

def train():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained('gpt2')

    def preprocess_function(examples):
        return tokenizer(["".join(x) for x in examples['card']])

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    getCorpus('cards.csv')
    dataset = load_dataset('csv', data_files='corpus.csv', split='train')
    print(dataset)
    dataset = dataset.train_test_split(test_size=0.1)
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched = True,
        num_proc = 4,
        remove_columns=dataset['train'].column_names,
    )

    lm_dataset = tokenized_datasets.map(group_texts, batched = True, num_proc = 4,)

    print(lm_dataset['train'][0])

    train_args = TrainingArguments(
        output_dir = 'magic_model',
        learning_rate = 2e-5,
        weight_decay = 0.01,
        push_to_hub = False,
        use_cpu = False,
        num_train_epochs = num_epochs
    )

    trainer = Trainer(
        model = model,
        args = train_args,
        train_dataset=lm_dataset['train'],
        eval_dataset=lm_dataset['test'],
        data_collator=data_collator,
    )

    trainer.train()


def gen():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('magic_model/checkpoint-6000')
    text = "<tl>"
    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    model.to(device)
    output = model.generate(
        **encoded_input,
        do_sample=True,
        temperature = 0.9,
        max_length =400,
    )
    print(tokenizer.batch_decode(output)[0])

if __name__ == '__main__':
    gen()