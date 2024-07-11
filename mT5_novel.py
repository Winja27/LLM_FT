import json
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

# 加载模型和分词器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = "./model/mt5-summary-model0"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

# 文本摘要函数
def summarize_text(text):
    # 编码输入文本
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)

    # 生成摘要
    summary_ids = model.generate(
        input_ids,
        max_length=1000,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # 解码生成的摘要
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    summary = summary.replace('<extra_id_0>', '')
    return summary

# 读取 JSON 文件
input_file = 'huozhe_cut.json'
output_file = 'output.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 对每个 text 标签生成摘要，去除换行符
for item in data:
    text = item['text'].replace('\n', '')
    summary = summarize_text(text)
    item['summary'] = summary

# 保存结果到新的 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("摘要生成完成并保存到 output.json 文件中。")
