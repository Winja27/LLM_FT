import torch
from transformers import BertTokenizer, EncoderDecoderModel

# 加载微调后的模型和tokenizer
model_path = "./model/bert-base-chinese-summary-model0"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = EncoderDecoderModel.from_pretrained(model_path)

# 检查GPU是否可用，将模型放在GPU上进行推理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# 设置decoder_start_token_id为[CLS]标记的ID
model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("[CLS]")

# 准备输入文本并生成摘要
input_text = ("澳大利亚央行将利率降至纪录低点,以应对疲软的经济前景,并遏制澳元进一步走强。05/0513:37|评论(0)A+澳大利亚央行周二发布声明称,将关键利率由2.25%调降至2%,"
              "符合此前交易员及接受彭博调查的29位经济学家中25位的预期。据彭博社报道,上月澳央行官员曾警告,矿业之外的行业投资可能下滑。澳大利亚政府不太可能推出新的刺激措施,"
              "来扶助受本币升值和铁矿石价格下跌打击而低于潜在水平的经济增长。“鉴于大宗商品价格下跌,矿业投资还可能有低于当前预期的风险,"
              "”预计到降息的澳新银行高级经济学家FelicityEmmett在决议公布前编写的研究报告中称。他表示此次决议可能反映出“央行经济增长预估轨迹有所下调”。")
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