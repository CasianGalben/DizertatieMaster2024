import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import pickle
import os


model_save_path = os.path.join('static', 'NER_MODEL.pth')


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='unicode_escape')
    df = df.head(10000)
    pos_counts = df['POS'].value_counts()
    minor_pos = pos_counts[pos_counts < 1000].index
    df['POS'] = df['POS'].apply(lambda x: 'OTHER' if x in minor_pos else x)
    df = df.astype({'Sentence #': 'string', 'Word': 'string', 'Tag': 'string'})
    return df


def build_vocabulary(df, column):
    items = set(df[column].fillna('UNK').values)
    vocab = {item: idx for idx, item in enumerate(items, start=1)}
    vocab['UNK'] = 0
    return vocab


def items_to_indices(df, vocab_word, vocab_tag):
    df['Word_idx'] = df['Word'].apply(lambda x: vocab_word.get(x, vocab_word['UNK']))
    df['Tag_idx'] = df['Tag'].apply(lambda x: vocab_tag.get(x, vocab_tag['UNK']))
    return df


def create_sequences(df):
    sequences = []
    temp_tokens, temp_tags = [], []
    for _, row in df.iterrows():
        if pd.isna(row['Sentence #']) or 'Sentence' in row['Sentence #']:
            if temp_tokens:
                sequences.append({'tokens': temp_tokens, 'tags': temp_tags})
                temp_tokens, temp_tags = [], []
        temp_tokens.append(row['Word_idx'])
        temp_tags.append(row['Tag_idx'])
    return sequences


class NERDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return torch.tensor(self.sequences[index]['tokens']), torch.tensor(self.sequences[index]['tags'])


class SimpleNLPModel(torch.nn.Module):
    def __init__(self, vocab_size, num_tags):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=128)
        self.lstm = torch.nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = torch.nn.Linear(256 * 2, num_tags)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.classifier(x)
        return x


def train_model(model, data_loader, epochs=10, learning_rate=0.001):
    training_details = {'losses': [], 'epochs': epochs}
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for words, tags in data_loader:
            optimizer.zero_grad()
            outputs = model(words)
            loss = criterion(outputs.view(-1, model.classifier.out_features), tags.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        training_details['losses'].append(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}')
    return training_details

def evaluate_model(model, data_loader, training_details):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for words, tags in data_loader:
            outputs = model(words)
            predicted = outputs.argmax(dim=2)
            total += tags.numel()
            correct += (predicted == tags).sum().item()
    accuracy = 100 * correct / total
    training_details['accuracy'] = f'Accuracy: {accuracy:.2f}%'
    return training_details


def save_model(model, path):
    torch.save(model.state_dict(), path)


def run_training(file_path):
    df = load_and_preprocess_data(file_path)
    vocab_word = build_vocabulary(df, 'Word')
    vocab_tag = build_vocabulary(df, 'Tag')
    df = items_to_indices(df, vocab_word, vocab_tag)
    sequences = create_sequences(df)
    train_seq, valid_seq = sequences[:int(0.85*len(sequences))], sequences[int(0.85*len(sequences)):]
    train_data = NERDataset(train_seq)
    valid_data = NERDataset(valid_seq)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False)
    model = SimpleNLPModel(len(vocab_word), len(vocab_tag))
    training_details = train_model(model, train_loader, epochs=20)
    training_details = evaluate_model(model, valid_loader, training_details)
    save_model(model, model_save_path)
    return training_details
