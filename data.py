import json

# 示例数据
train_data = [
    {
        "text": "在一个风雨交加的夜晚，年轻的侦探约翰·史密斯踏入了被称为幽灵屋的古老庄园。屋内，一切看似平常，但约翰能感觉到一股不寻常的气息。他的直觉告诉他，这里发生过些什么。墙上的旧画像，仿佛在诉说着过去的秘密，而每一道门后，都可能隐藏着一个故事。约翰小心翼翼地走过长长的走廊，他的脚步声在空旷的房间中回响。",
        "summary": "第一章介绍了主人公约翰·史密斯和神秘的幽灵屋。"
    }
]

validation_data = [
    {
        "text": "在揭开最后一层面纱时，约翰意识到真相远比他想象的要复杂。庄园内的每个人都有可能是幕后黑手。管家的神秘失踪，女仆的窃窃私语，以及邻居的不寻常兴趣，所有这些都指向了一个令人震惊的结论。在一场突如其来的火灾中，约翰找到了最后的线索，揭示了艾莉森的真实身份和她与庄园主人之间的秘密关系。",
        "summary": "第三章达到故事高潮，揭示了庄园内每个人都可能涉案。"
    },
]

# 生成大量数据
train_data_large = train_data * 1000
validation_data_large = validation_data * 100

# 保存到JSON文件
with open('train1.json', 'w', encoding='utf-8') as f:
    json.dump(train_data_large, f, ensure_ascii=False, indent=4)

with open('validation1.json', 'w', encoding='utf-8') as f:
    json.dump(validation_data_large, f, ensure_ascii=False, indent=4)
