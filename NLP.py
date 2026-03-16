from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 1. Load your custom labeled dataset
dataset = load_dataset("csv", data_files="cleaned.csv")

# 2. Load the base FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)

# 3. Tokenize the text (convert words to numbers for the neural network)
def tokenize_function(examples):
    return tokenizer(examples["Text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Set up the Training Loop
training_args = TrainingArguments(
    output_dir="./custom_finbert",
    num_train_epochs=3,              # Pass over the data 3 times
    per_device_train_batch_size=16,  # Process 16 headlines at a time
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# 5. TRAIN THE MODEL!
trainer.train()

# 6. Save your newly trained, custom model to your computer
trainer.save_model("./my-custom-finbert")