import pandas as pd
from datasets import Dataset
from transformers import BertForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import random

# Step 1: Load the dataset
file_path = "corpus.csv"
df = pd.read_csv(file_path)

# Check the first few rows
print(df.head())

# Step 2: Preprocessing the dataset
# Mask tokens for MLM
def mask_tokens(example):
    tokens = example.split()  # Tokenize by whitespace
    masked_text = []
    for token in tokens:
        if random.random() < 0.15:  # Mask 15% of tokens
            masked_text.append("[MASK]")
        else:
            masked_text.append(token)
    return {"masked_text": " ".join(masked_text), "original_text": example}

# Create a Hugging Face dataset
dataset = Dataset.from_pandas(df.rename(columns={"card": "text"}))  # Rename for consistency

# Apply masking
dataset = dataset.map(lambda x: mask_tokens(x["text"]))

# Step 3: Tokenize the dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["masked_text"], truncation=True, padding="max_length", max_length=200)

tokenized_dataset = dataset.map(tokenize_function, batched=True)



dataset_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
test_dataset = dataset_split["test"]

# Step 5: Data Collator for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Step 6: Load the Model and Trainer
model = BertForMaskedLM.from_pretrained("bert-base-uncased")



training_args = TrainingArguments(
    output_dir="./bert-fine-tuned-mtg-5",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    logging_first_step=True,
    disable_tqdm=False,

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 7: Train the model
trainer.train(resume_from_checkpoint="./bert-fine-tuned-mtg-5/checkpoint-15470")
