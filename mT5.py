import json
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "./model/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

with open('train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('validation.json', 'r', encoding='utf-8') as f:
    validation_data = json.load(f)

dataset = DatasetDict({
    'train': Dataset.from_list(train_data),
    'validation': Dataset.from_list(validation_data)
})

def preprocess_function(examples):
    inputs = examples['text']
    targets = examples['summary']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

sample_data = {
    "text": "在一个风雨交加的夜晚，年轻的侦探约翰·史密斯踏入了被称为幽灵屋的古老庄园。",
    "summary": "第一章介绍了主人公约翰·史密斯和神秘的幽灵屋。"
}
processed_data = preprocess_function(sample_data)
print(processed_data)
tokenized_datasets = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=1,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

trainer.train()

model.save_pretrained("./model/mt5-summary-model")
tokenizer.save_pretrained("./model/mt5-summary-model")
