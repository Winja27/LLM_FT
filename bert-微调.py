import torch
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "./model/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name).to(device)


# 计算模型参数总数的函数
def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 打印模型参数总数
total_params = count_model_parameters(model)
print(f'Total parameters: {total_params}')


# 读取JSON文件并创建Dataset对象
def load_json_file_to_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data_dicts = json.load(f)
    return Dataset.from_dict({
        "text": [d['text'] for d in data_dicts],
        "summary": [d['summary'] for d in data_dicts]
    })


train_dataset = load_json_file_to_dataset("训练数据（新）.json")
validation_dataset = load_json_file_to_dataset("validation.json")


# 定义预处理函数
def preprocess_function(examples):
    inputs = examples['text']
    targets = examples['summary']

    global tokenizer, device

    # 添加 attention_mask
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    bos_token_id = tokenizer.bos_token_id
    decoder_input_ids = torch.full((labels.input_ids.shape[0], 128),
                                   bos_token_id if bos_token_id is not None else tokenizer.pad_token_id)

    model_inputs["decoder_input_ids"] = decoder_input_ids
    model_inputs["labels"] = labels.input_ids
    model_inputs["attention_mask"] = model_inputs["attention_mask"]  # 添加 attention_mask
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]  # 添加 decoder_attention_mask

    labels.input_ids[labels.input_ids == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels.input_ids

    return model_inputs


# 对训练集和验证集进行预处理
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_validation_dataset = validation_dataset.map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./model/bert-base-chinese-summary-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=12,
    weight_decay=0.01
)

# 创建Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
)

# 训练模型
trainer.train()

# 保存模型和分词器
model.save_pretrained("./model/bert-base-chinese-summary-model")
tokenizer.save_pretrained("./model/bert-base-chinese-summary-model")
