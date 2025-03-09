import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import os

# Constants
FBLTHP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FBLTHP_DIR, 'data')
DATASET_DIR = os.path.join(DATA_DIR, 'datasets')
DEFAULT_LINES_PATH = os.path.join(DATASET_DIR, 'lines.txt')
MOMIR_LINES_PATH = os.path.join(DATASET_DIR, 'momir_lines.txt')


def train_on_text_file(
    txt_file_path, 
    output_dir='text_model',
    num_epochs=3,
    block_size=125,
    batch_size=8,
    gradient_accumulation_steps=4,
    fp16=True
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize the model
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.to(device)
    
    # Load and preprocess text data
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Create a Dataset
    dataset_dict = {"text": lines}
    dataset = Dataset.from_dict(dataset_dict)
    
    # Tokenization function
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=block_size,
            return_tensors=None
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    # Apply tokenization with parallelization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
        num_proc=max(1, os.cpu_count() // 2)
    )
    
    # Split into train and test
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=100,
        push_to_hub=False,
        fp16=fp16 and device == 'cuda',
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=2 if device == 'cuda' else 0,
        save_total_limit=1,
        report_to="none",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(f"./{output_dir}/final")
    tokenizer.save_pretrained(f"./{output_dir}/final")


# Function to generate text
def generate_text(
    prompt_text,
    model_path='./text_model/final',
    max_length=150,
    temperature=0.9,
    samples=10
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    
    # Encode input
    encoded_input = tokenizer(prompt_text, return_tensors='pt').to(device)
    returned = []
    for i in range(samples):
        # Generate output
        output = model.generate(
            **encoded_input,
            do_sample=True,
            temperature=temperature,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.92,
            top_k=50
        )
        returned.append(tokenizer.batch_decode(output, skip_special_tokens=True)[0])
    
    # Decode and return the result
    return returned

if __name__ == '__main__':
    # Train the model
    train_on_text_file(
        MOMIR_LINES_PATH, 
        output_dir='momir',
        block_size=250,
        )
    
    # Generate sample text
    print("\nGenerating sample output...")
    samples = generate_text("<mc>", samples=5)
    for s in samples:
        print(s)