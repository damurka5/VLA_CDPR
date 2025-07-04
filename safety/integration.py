from transformers import TrainingArguments, Trainer
from datasets import Dataset

def train_safety_classifier():
    # Load generated dataset
    with open("cdpr_safety_dataset.json") as f:
        data = json.load(f)
    
    # Convert to HuggingFace dataset
    hf_dataset = Dataset.from_list(data)
    dataset = hf_dataset.train_test_split(test_size=0.2)
    
    # Load tokenizer and model
    model_name = "bert-base-uncased"  # Good starting point
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["prompt"], padding="max_length", truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Convert labels
    def map_labels(examples):
        examples["label"] = 1 if examples["safety_label"] == "safe" else 0
        return examples
    
    labeled_dataset = tokenized_dataset.map(map_labels)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./safety_classifier",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=labeled_dataset["train"],
        eval_dataset=labeled_dataset["test"],
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model("./safety_classifier/final_model")

if __name__ == "__main__":
    train_safety_classifier()