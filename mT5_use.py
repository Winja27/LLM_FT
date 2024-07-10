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
    summary_ids = model.generate(
        input_ids,
        max_length=1000,
        #min_length=10,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # 解码生成的摘要
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    summary = summary.replace('<extra_id_0>', '')
    return summary


# 调用文本摘要函数
text_to_summarize = ("""发布日期:2015-07-1215:20:00滁州市气象台2015年07月12日15时20分发布雷电黄色预警信号:目前我市西部有较强对流云团向东南方向移动,预计6小时内我市部分地区将发生雷电活动,并可能伴有短时强降水、大风、局部冰雹等强对流天气,请注意防范。图例标准防御指南6小时内可能发生雷电活动,可能会造成雷电灾害事故。1、政府及相关部门按照职责做好防雷工作;2、密切关注天气,尽量避免户外活动。""")
print(summarize_text(text_to_summarize))
