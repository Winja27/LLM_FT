import torch
from torch import nn
from transformers import GPT2Tokenizer

# 加载预训练的GPT2 tokenizer，并指定填充符和其他特殊标记
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # 使用eos_token作为填充符
tokenizer.bos_token = tokenizer.eos_token  # 使用eos_token作为开始符
tokenizer.eos_token_id = tokenizer.eos_token_id

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
def generate_summary(model, tokenizer, text, max_length=100):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        print(f"Input text: {text}")
        print(f"Input IDs: {input_ids}")

        summary_ids = torch.full((1, max_length), tokenizer.pad_token_id, dtype=torch.long).to(device)
        summary_ids[0, 0] = tokenizer.eos_token_id  # 使用eos_token_id作为开始符

        for i in range(1, max_length):
            outputs = model(input_ids, summary_ids[:, :i])
            next_token_logits = outputs[0, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1).item()
            summary_ids[0, i] = next_token_id

            # 打印中间结果进行调试
            print(f"Step {i}, Next token id: {next_token_id}, Token: {tokenizer.decode([next_token_id])}")
            if next_token_id == tokenizer.eos_token_id:  # Assuming EOS token is used to end the sequence
                break

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 使用示例文本生成摘要
text = "第一章我比现在年轻十岁的时候，获得了一个游手好闲的职业，去乡间收集民间歌谣。那一年的整个夏天，我如同一只乱飞的麻雀，游荡在知了和阳光充斥的村舍田野。我喜欢喝农民那种带有苦味的茶水，他们的茶桶就放在田埂的树下，我毫无顾忌地拿起漆满茶垢的茶碗舀水喝，还把自己的水壶灌满，与田里干活的男人说上几句废话，在姑娘因我而起的窃窃私笑里扬长而去。我曾经和一位守着瓜田的老人聊了整整一个下午，这是我有生以来瓜吃得最多的一次，当我站起来告辞时，突然发现自己像个孕妇一样步履艰难了。然后我与一位当上了祖母的女人坐在门槛上，她编着草鞋为我唱了一支《十月怀胎》。我最喜欢的是傍晚来到时，坐在农民的屋前，看着他们将提上的井水泼在地上，压住蒸腾的尘土，夕阳的光芒在树梢上照射下来，拿一把他们递过来的扇子，尝尝他们和盐一样咸的咸菜，看看几个年轻女人，和男人们说着话。我头戴宽边草帽，脚上穿着拖鞋，一条毛巾挂在身后的皮带上，让它像尾巴似的拍打着我的屁股。我整日张大嘴巴打着呵欠，散漫地走在田间小道上，我的拖鞋吧哒吧哒，把那些小道弄得尘土飞扬，仿佛是车轮滚滚而过时的情景。我到处游荡，已经弄不清楚哪些村庄我曾经去过，哪些我没有去过。我走近一个村子时，常会听到孩子的喊叫：“那个老打呵欠的人又来啦。”于是村里人就知道那个会讲荤故事会唱酸曲的人又来了。其实所有的荤故事所有的酸曲都是从他们那里学来的，我知道他们全部的兴趣在什么地方，自然这也是我的兴趣。"
summary = generate_summary(model, tokenizer, text)
print("摘要：" + summary)
