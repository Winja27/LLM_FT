import torch
from torch import nn
from transformers import AutoTokenizer

# 加载预训练的tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


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


# 加载训练好的模型
input_dim = tokenizer.vocab_size
output_dim = tokenizer.vocab_size
model = TransformerModel(input_dim, output_dim)
model.load_state_dict(torch.load('./model/transformer-model.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()


# 生成摘要函数
def generate_summary(model, tokenizer, text, max_length=50):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        summary_ids = torch.full((1, max_length), tokenizer.pad_token_id, dtype=torch.long).to(device)
        summary_ids[0, 0] = tokenizer.cls_token_id  # Assuming CLS token is used to start the sequence

        for i in range(1, max_length):
            outputs = model(input_ids, summary_ids[:, :i])
            next_token_logits = outputs[0, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1).item()
            summary_ids[0, i] = next_token_id
            if next_token_id == tokenizer.sep_token_id:  # Assuming SEP token is used to end the sequence
                break

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# 使用示例文本生成摘要
text = "在一个风雨交加的夜晚，年轻的侦探约翰·史密斯踏入了被称为幽灵屋的古老庄园。屋内，一切看似平常，但约翰能感觉到一股不寻常的气息。他的直觉告诉他，这里发生过些什么。墙上的旧画像，仿佛在诉说着过去的秘密，而每一道门后，都可能隐藏着一个故事。约翰小心翼翼地走过长长的走廊，他的脚步声在空旷的房间中回响。"

summary = generate_summary(model, tokenizer, text)
print("摘要："+summary)
