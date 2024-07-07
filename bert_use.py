import torch
from transformers import BertTokenizer, EncoderDecoderModel

# 加载微调后的模型和tokenizer
model_path = "./model/bert-base-chinese-summary-model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = EncoderDecoderModel.from_pretrained(model_path)

# 检查GPU是否可用，将模型放在GPU上进行推理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# 设置decoder_start_token_id为[CLS]标记的ID
model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("[CLS]")

# 准备输入文本并生成摘要
input_text = ("在一个风雨交加的夜晚，年轻的侦探约翰·史密斯踏入了被称为幽灵屋的古老庄园。屋内，一切看似平常，但约翰能感觉到一股不寻常的气息。他的直觉告诉他，这里发生过些什么。墙上的旧画像，仿佛在诉说着过去的秘密，而每一道门后，都可能隐藏着一个故事。约翰小心翼翼地走过长长的走廊，他的脚步声在空旷的房间中回响。")
inputs = tokenizer(input_text, max_length=128, truncation=True, padding="max_length", return_tensors="pt").to(device)

# 使用模型生成摘要
summary_ids = model.generate(
    inputs.input_ids,
    decoder_start_token_id=model.config.decoder_start_token_id,  # 直接作为参数传递
    max_length=50,
    num_beams=4,
    length_penalty=2.0,
    early_stopping=True
)

# 解码生成的摘要
generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Generated Summary:", generated_summary)