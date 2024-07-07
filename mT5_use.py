import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

# 加载模型和分词器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = "./model/mt5-summary-model"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)


# 文本摘要函数
def summarize_text(text):
    # 编码输入文本
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)

    # 生成摘要
    summary_ids = model.generate(input_ids, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

    # 解码生成的摘要
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


# 调用文本摘要函数
text_to_summarize = "随着调查的深入，约翰发现了一系列令人不安的线索，似乎指向了庄园主人的突然失踪与一个古老的家族诅咒有关。在图书馆的一本破旧日记中，他找到了一段模糊的记录，提到了一个禁忌的仪式和一个名叫艾莉森的女子。约翰意识到，这个艾莉森可能是解开谜团的关键。"
print(summarize_text(text_to_summarize))
