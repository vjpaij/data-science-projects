import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom Dataset
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Model Architecture
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, model_name='bert-base-multilingual-cased'):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        output = self.drop(pooled_output)
        return self.out(output)

# Training setup
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / n_examples, np.mean(losses)

# Evaluation function
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)

# Main training loop
def train_pytorch_model(df, lang='en'):
    # Prepare data
    lang_df = df[df['detected_language'] == lang].copy()
    lang_df['sentiment_label'] = lang_df['review_rating'].apply(
        lambda x: 1 if x > 3 else 0)
    lang_df = lang_df[lang_df['sentiment_label'].isin([0, 1])]
    
    # Split data
    df_train, df_test = train_test_split(
        lang_df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(
        df_test, test_size=0.5, random_state=42)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Create datasets
    MAX_LEN = 128
    BATCH_SIZE = 16
    
    train_dataset = ReviewDataset(
        df_train['processed_text'].to_numpy(),
        df_train['sentiment_label'].to_numpy(),
        tokenizer,
        MAX_LEN
    )
    
    val_dataset = ReviewDataset(
        df_val['processed_text'].to_numpy(),
        df_val['sentiment_label'].to_numpy(),
        tokenizer,
        MAX_LEN
    )
    
    test_dataset = ReviewDataset(
        df_test['processed_text'].to_numpy(),
        df_test['sentiment_label'].to_numpy(),
        tokenizer,
        MAX_LEN
    )
    
    # Create data loaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )
    
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )
    
    # Initialize model
    model = SentimentClassifier(n_classes=2)
    model = model.to(device)
    
    # Training parameters
    EPOCHS = 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        total_iters=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    # Training loop
    history = defaultdict(list)
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        
        print(f'Train loss {train_loss} accuracy {train_acc}')
        
        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )
        
        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()
        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), f'models/best_model_{lang}.bin')
            best_accuracy = val_acc
    
    # Evaluate on test set
    test_acc, test_loss = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(df_test)
    )
    
    print(f'Test accuracy: {test_acc.item()}')
    
    return model, history

# Train for English
pytorch_model, history = train_pytorch_model(df, 'en')