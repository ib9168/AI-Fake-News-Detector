# This script converts the CSV data into a format PyTorch understands, trains the BERT model,
# and saves the intelligent version.


import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os

# 1. Custom Dataset Class (Required for PyTorch)
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(path, tokenizer):
    df = pd.read_csv(path)
    # Ensure dataset isn't too massive for local testing; limit to 1000 rows if needed
    # df = df.head(1000) 
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return NewsDataset(encodings, labels)

def main():
  
    # Gets the directory where this script (train_model.py) is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Constructs absolute paths get to the data files
    train_path = os.path.join(BASE_DIR, 'data', 'train.csv')
    val_path = os.path.join(BASE_DIR, 'data', 'val.csv')
    save_path = os.path.join(BASE_DIR, 'saved_model')
    output_dir = os.path.join(BASE_DIR, 'results')
    logs_dir = os.path.join(BASE_DIR, 'logs')
    
    print("Loading Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print(f"Loading Data from: {train_path}")
    
    # Checks if files exist before crashing
    if not os.path.exists(train_path):
        print(f"ERROR: File not found at {train_path}")
        return

    train_dataset = load_data(train_path, tokenizer)
    val_dataset = load_data(val_path, tokenizer)

    print("Loading Model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir=output_dir,          # Updated path
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logs_dir,           # Updated path
        logging_steps=10,
        eval_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("Starting Training (Fine-tuning)...")
    trainer.train()

    print("Saving Model...")
    # Creates directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()