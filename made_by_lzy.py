import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from transformers import GPT2Tokenizer

# 加载预训练的GPT2 tokenizer，并指定填充符
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # 使用eos_token作为填充符
tokenizer.bos_token = tokenizer.eos_token  # 使用eos_token作为开始符
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Transformer模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_embedding = nn.Embedding(input_dim, d_model)
        self.trg_embedding = nn.Embedding(output_dim, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout)
        self.fc_out = nn.Linear(d_model, output_dim)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg):
        src = self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        trg = self.trg_embedding(trg) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.dropout(src)
        trg = self.dropout(trg)
        src = src.permute(1, 0, 2)
        trg = trg.permute(1, 0, 2)
        output = self.transformer(src, trg)
        output = output.permute(1, 0, 2)
        output = self.fc_out(output)
        return output


# 数据集定义
class TextSummaryDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        summary = self.data[idx]['summary']
        return text, summary


# 数据加载器
def collate_fn(batch):
    texts, summaries = zip(*batch)
    text_encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    summary_encodings = tokenizer(summaries, padding=True, truncation=True, return_tensors='pt')
    return text_encodings.input_ids, summary_encodings.input_ids


train_dataset = TextSummaryDataset('train1.json')
val_dataset = TextSummaryDataset('validation1.json')

train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

# 模型实例化
input_dim = tokenizer.vocab_size
output_dim = tokenizer.vocab_size
model = TransformerModel(input_dim, output_dim)

# 损失函数和优化器
# 确保损失函数忽略填充的部分
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

optimizer = optim.Adam(model.parameters(), lr=0.00001)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')


# 训练与评估函数
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        # Shift trg for input and output
        output = model(src, trg[:, :-1])
        output_dim = output.shape[-1]

        # Flatten output and trg for loss calculation
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg[:, :-1])
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# 训练过程
N_EPOCHS = 3
CLIP = 1
model = model.to(device)

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_loader, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './model/transformer-model.pt')

    print(f'Epoch: {epoch + 1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')
