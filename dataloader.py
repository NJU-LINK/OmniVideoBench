import json
from typing import List,Dict,Any
import os

def convert_duration_to_seconds(time_str):
    """
    Converts a time string in 'MM:SS' or 'HH:MM:SS' format to total seconds.
        int: The total duration in seconds.
    """
    parts = time_str.split(':')
    seconds = 0
    if len(parts) == 2:  # MM:SS format
        minutes = int(parts[0])
        seconds = int(parts[1])
        total_seconds = minutes * 60 + seconds
    else:
        raise ValueError("Invalid time format. Please use 'MM:SS' or 'HH:MM:SS'.")
    
    return total_seconds

class VideoQADaloader:
    """Load our data for batch-size evaluation."""
    def __init__(self,data_json_file:str,video_dir:str):
        """Init the dataloader."""
        self.data_json_file=data_json_file
        self.data=self._load_data()
        self.video_dir=video_dir

    def _load_data(self)-> List[Dict[str,Any]]:
        """Load and extract data."""
        try:
            with open(self.data_json_file,'r',encoding='utf-8')as f:
                data=json.load(f)
                print(f'Succeed to loading data from {self.data_json_file}')
                return data
        except FileNotFoundError:
            print(f"File {self.data_json_file} can't be found!")
            return []
        except json.JSONDecodeError:
            print(f"Json file {self.data_json_file} can't be decoded!")
            return []
        except Exception as e:
            print(f'Some unknown error happens when loading data:{e}!')
            return []

    def get_all_qa_pairs(self)-> List[Dict[str,Any]]:
        extracted_data=[]
        for item in self.data:
            video_path=os.path.join(self.video_dir,item.get('video','unknown_video')+".mp4")
            duration=convert_duration_to_seconds(item.get('duration'))
            for qa in item.get('questions',[]):
                test_item={
                    'video_path':video_path,
                    'duration':duration,
                    'question':qa.get('question'),
                    'options':qa.get('options'),
                    'answer':qa.get('correct_option')
                }
                extracted_data.append(test_item)
            # print(len(extracted_data))
        return extracted_data
    
if __name__=="__main__":

    # Test the dataset class
    data_json_file="data/merged_qas_1_0817.json"
    video_dir="/omni_videos_v1/10"
    dataloader=VideoQADaloader(data_json_file,video_dir)

    if dataloader.data:
        all_qa_data=dataloader.get_all_qa_pairs()
        print(f'Succeed to extract {len(all_qa_data)} question-answer pairs.')
        if all_qa_data:
            print(json.dumps(all_qa_data[0], indent=2, ensure_ascii=False))

        