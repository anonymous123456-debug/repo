import json

input_json_path = "./mcqa.json"  # 你的JSON文件路径
output_txt_path = "./mcqa.txt"  # 生成的纯文本文件

with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

all_texts = []

for item in data:
    if "text" in item:
        all_texts.append(item["text"])

# 拼接所有 text 字段，用换行隔开
combined_text = "\n".join(all_texts)

with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write(combined_text)

print(f"已成功将所有 text 字段内容拼接并保存到 {output_txt_path}")
