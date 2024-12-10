import pandas as pd
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Step 1: Load the dataset
file_path = "corpus.csv"
df = pd.read_csv(file_path)

# Check the first few rows
print(df.head())

# Step 2: Preprocessing the dataset
# GPT-2 doesn't require explicit token masking for MLM as it is an autoregressive model.
# So we'll directly use the "text" column (or whatever column contains your text data).

# Create a Hugging Face dataset
dataset = Dataset.from_pandas(df.rename(columns={"card": "text"}))  # Rename for consistency

# Step 3: Tokenize the dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2 requires padding on the right and no special padding tokens by default
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=200)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 4: Split the dataset
dataset_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
test_dataset = dataset_split["test"]

# Step 5: Data Collator for Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # GPT-2 is not trained for masked language modeling
)

# Step 6: Load the GPT-2 Model and Trainer
model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2-fine-tuned-mtg",       # Directory to save model checkpoints
    evaluation_strategy="epoch",             # Evaluate at the end of each epoch
    learning_rate=5e-5,                      # Learning rate for fine-tuning
    per_device_train_batch_size=16,          # Batch size per device (GPU/CPU)
    num_train_epochs=20,                     # Number of epochs
    weight_decay=0.01,                       # Regularization via weight decay
    save_strategy="epoch",                   # Save checkpoints at each epoch
    logging_dir="./logs",                    # Directory for logs
    logging_steps=50,                        # Log every 50 steps
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
trainer.train()

# Save the final fine-tuned model
model.save_pretrained("./gpt2-fine-tuned-mtg")
tokenizer.save_pretrained("./gpt2-fine-tuned-mtg")