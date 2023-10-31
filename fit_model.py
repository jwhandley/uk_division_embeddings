import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")

# 1. Model Definition
class VotePredictor(nn.Module):
    def __init__(self, num_bills, num_legislators, latent_dim=2):
        super(VotePredictor, self).__init__()
        
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.legislator_bias = nn.Embedding(num_legislators, 1)
        self.bill_bias = nn.Embedding(num_bills, 1)
        
        self.legislator_embedding = nn.Embedding(num_legislators, latent_dim)
        self.bill_embedding = nn.Embedding(num_bills, latent_dim)

    def forward(self, bill_ids, legislator_ids):
        global_bias = self.global_bias
        legislator_bias = self.legislator_bias(legislator_ids).squeeze()
        bill_bias = self.bill_bias(bill_ids).squeeze()
        
        legislator_embedding = self.legislator_embedding(legislator_ids)
        bill_embedding = self.bill_embedding(bill_ids)
        
        interaction = (legislator_embedding * bill_embedding).sum(1)
        logits = global_bias + legislator_bias + bill_bias + interaction
        return torch.sigmoid(logits)

# 2. Dataset and DataLoader
class VotingDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        bill_id, legislator_id, vote = self.data[idx]
        return bill_id, legislator_id, vote

logging.info("Loading data")
# with open('data/divisions.json') as f:
#     data = json.load(f)

data = pd.read_csv('data/H117_votes_encoded.csv').values

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data[:, 2])
train_dataset = VotingDataset(train_data)
test_dataset = VotingDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

logging.info("Training model")
num_bills = max([x[0] for x in data]) + 1
logging.info(f"Number of bills: {num_bills}")
num_legislators = max([x[1] for x in data]) + 1
logging.info(f"Number of legislators: {num_legislators}")


model = VotePredictor(num_bills, num_legislators).to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epoch_losses = []
num_epochs = 40
for epoch in tqdm(range(num_epochs), desc="Epochs", leave=False):
    model.train()
    
    for bill_ids, legislator_ids, votes in tqdm(train_loader, desc="Batches", leave=False):
        optimizer.zero_grad()
        
        bill_ids, legislator_ids, votes = bill_ids.to(device), legislator_ids.to(device), votes.float().to(device)
        outputs = model(bill_ids, legislator_ids)
        loss = criterion(outputs, votes)
        
        loss.backward()
        optimizer.step()

    model.eval()
    test_loss = 0.0
    test_f1 = 0.0
    test_batches = 0

    with torch.no_grad():
        for bill_ids, legislator_ids, votes in tqdm(test_loader, desc="Batches", leave=False):
            bill_ids, legislator_ids, votes = bill_ids.to(device), legislator_ids.to(device), votes.float().to(device)
            outputs = model(bill_ids, legislator_ids)
            loss = criterion(outputs, votes)
            test_loss += loss.item()
            
            binary_preds = (outputs > 0.5).float()
            f1 = f1_score(votes.cpu().numpy(), binary_preds.cpu().detach().numpy())
            test_f1 += f1
            test_batches += 1

    avg_test_loss = test_loss / test_batches
    avg_test_f1 = test_f1 / test_batches
    logging.info(f"Test Loss: {avg_test_loss:.4f}, Test F1 Score: {avg_test_f1:.4f}")

legislator_embeddings = model.legislator_embedding.weight.detach().cpu().numpy()
bill_embeddings = model.bill_embedding.weight.detach().cpu().numpy()

logging.info("Saving embeddings")
np.save("output/legislator_embeddings.npy", legislator_embeddings)
np.save("output/bill_embeddings.npy", bill_embeddings)