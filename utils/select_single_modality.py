import json
import argparse

# 默认输入输出路径 - 使用相对路径
DEFAULT_INPUT_JSON = "data/input.json"
DEFAULT_OUTPUT_JSON = "data/output_single_modality.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def filter_questions(data):
    filtered_data = []

    for video in data:
        new_video = video.copy()
        new_questions = []
        print(video["video"])

        for q in video.get("questions", []):
            steps = q.get("reasoning_steps", [])
            modalities = {step["modality"] for step in steps if "modality" in step}

            # 判断是否只有一个模态
            if len(modalities) == 1:
                new_questions.append(q)

        if new_questions:
            new_video["questions"] = new_questions
            filtered_data.append(new_video)

    return filtered_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="筛选单模态问题")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT_JSON, 
                       help="输入JSON文件路径")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_JSON,
                       help="输出JSON文件路径")
    args = parser.parse_args()
    
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    
    data = load_json(args.input)
    filtered = filter_questions(data)
    save_json(filtered, args.output)
    print(f"筛选完成，结果已保存到 {args.output}")
