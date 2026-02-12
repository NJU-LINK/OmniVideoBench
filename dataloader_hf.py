import os
import json
from typing import List, Dict, Any, Optional
from datasets import load_dataset

""""
# Hugging Face (default)
python OmniVideoBench/dataloader_hf.py --video_dir ./videos

# Local file
python OmniVideoBench/dataloader_hf.py --data_json_file data.json --video_dir ./videos
"""


def convert_duration_to_seconds(time_str):
    parts = time_str.split(':')
    seconds = 0
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = int(parts[1])
        total_seconds = minutes * 60 + seconds
    else:
        raise ValueError("Invalid time format. Please use 'MM:SS' or 'HH:MM:SS'.")
    
    return total_seconds


class VideoQADaloader:
    def __init__(self, data_json_file: Optional[str] = None, video_dir: Optional[str] = None, hf_cache_dir: Optional[str] = None):
        self.data_json_file = data_json_file
        self.video_dir = video_dir
        self.hf_cache_dir = hf_cache_dir
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        if self.data_json_file and self.data_json_file.startswith("hf://"):
            try:
                ds = load_dataset(
                    self.data_json_file.replace("hf://", ""),
                    cache_dir=self.hf_cache_dir,
                    trust_remote_code=True
                )
                data = ds['test'].to_list()
                print(f'Succeed to load data from Hugging Face: {len(data)} items')
                return data
            except Exception as e:
                print(f'Error loading data from Hugging Face: {e}')
                return []
        elif self.data_json_file:
            try:
                with open(self.data_json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f'Succeed to loading data from {self.data_json_file}')
                    return data
            except Exception as e:
                print(f'Error loading data from {self.data_json_file}: {e}')
                return []
        else:
            try:
                ds = load_dataset(
                    "NJU-LINK/OmniVideoBench",
                    cache_dir=self.hf_cache_dir,
                    trust_remote_code=True
                )
                data = ds['test'].to_list()
                print(f'Succeed to load data from Hugging Face: {len(data)} items')
                return data
            except Exception as e:
                print(f'Error loading data from Hugging Face: {e}')
                return []
    
    def get_all_qa_pairs(self) -> List[Dict[str, Any]]:
        if not self.data:
            return []
        
        extracted_data = []
        for item in self.data:
            video_name = item.get('video', 'unknown_video')
            
            if self.video_dir:
                video_path = os.path.join(self.video_dir, video_name + ".mp4")
            else:
                video_path = os.path.join(self.video_dir or "", video_name + ".mp4") if self.video_dir else None
            
            duration = convert_duration_to_seconds(item.get('duration'))
            
            for qa in item.get('questions', []):
                test_item = {
                    'video_path': video_path,
                    'duration': duration,
                    'question': qa.get('question'),
                    'options': qa.get('options'),
                    'answer': qa.get('correct_option')
                }
                extracted_data.append(test_item)
        
        print(f'Extracted {len(extracted_data)} QA pairs')
        return extracted_data
    
    def download_videos(self, output_dir: str, num_videos: int = 10):
        ds = load_dataset(
            "NJU-LINK/OmniVideoBench",
            split="test",
            data_dir="videos",
            cache_dir=self.hf_cache_dir or output_dir,
            trust_remote_code=True
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        video_count = 0
        for item in ds:
            video_name = item['video']
            video_path = os.path.join(output_dir, video_name + ".mp4")
            
            if not os.path.exists(video_path):
                print(f"Downloading {video_name}...")
                ds.download_item(video_name, dest=video_path)
                video_count += 1
                
                if video_count >= num_videos:
                    break
        
        print(f"Downloaded {video_count} videos to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OmniVideoBench DataLoader")
    parser.add_argument("--data_json_file", type=str, default=None,
                       help="Path to local data JSON file or 'hf://NJU-LINK/OmniVideoBench' for Hugging Face")
    parser.add_argument("--video_dir", type=str, default="./videos",
                       help="Directory containing video files")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Hugging Face cache directory")
    args = parser.parse_args()
    
    dataloader = VideoQADaloader(
        data_json_file=args.data_json_file,
        video_dir=args.video_dir,
        hf_cache_dir=args.cache_dir
    )
    
    if dataloader.data:
        all_qa_data = dataloader.get_all_qa_pairs()
        if all_qa_data:
            import json
            print(json.dumps(all_qa_data[0], indent=2, ensure_ascii=False))
