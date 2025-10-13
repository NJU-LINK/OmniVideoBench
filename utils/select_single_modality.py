import json
import argparse

# Default input/output paths - using relative paths
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

            # Check if there's only one modality
            if len(modalities) == 1:
                new_questions.append(q)

        if new_questions:
            new_video["questions"] = new_questions
            filtered_data.append(new_video)

    return filtered_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter single modality questions")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT_JSON,
                       help="Input JSON file path")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_JSON,
                       help="Output JSON file path")
    args = parser.parse_args()
    
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    
    data = load_json(args.input)
    filtered = filter_questions(data)
    save_json(filtered, args.output)
    print(f"Filtering completed, results saved to {args.output}")
