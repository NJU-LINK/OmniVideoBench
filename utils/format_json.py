import json
import sys

def format_json_file(input_file, output_file=None):
    """
    读取 JSON 文件，格式化后写入新文件或覆盖原文件。
    
    Args:
        input_file: 输入的 JSON 文件路径。
        output_file: 输出的 JSON 文件路径。如果为 None，则覆盖输入文件。
    """
    try:
        # 读取原始 JSON 文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 将 JSON 文件转换为 Python 对象 [[10]]
    except json.JSONDecodeError as e:
        print(f"读取或解析 JSON 文件时出错: {e}")
        return
    except FileNotFoundError:
        print(f"文件未找到: {input_file}")
        return

    # 确定输出文件
    output_path = output_file if output_file else input_file

    # 将 Python 对象以格式化的方式写入 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)  # 使用 indent 参数进行美化 [[5]]
    
    print(f"JSON 文件已成功格式化并保存至: {output_path}")

# 使用示例
if __name__ == "__main__":
    
    input_json_file = "data.json"
    output_json_file = "data_formatted.json"
    
    format_json_file(input_json_file, output_json_file)