import json
import os
import string
import time
import threading
import argparse
import subprocess
import tempfile
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from google import genai
from google.genai import types
from dataloader import VideoQADaloader


class ThreadSafeResultCollector:
    def __init__(self):
        self.results_list = []
        self.completed_count = 0
        self.correct_count = 0
        self.failed_count = 0
        self.lock = threading.Lock()
    
    def add_result(self, result: Dict[str, Any]):
        with self.lock:
            self.results_list.append(result)
            if result.get('error_message'):
                self.failed_count += 1
            else:
                self.completed_count += 1
                if result.get('is_correct', False):
                    self.correct_count += 1
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get a copy of all results"""
        with self.lock:
            return self.results_list.copy()
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics"""
        with self.lock:
            return {
                'completed': self.completed_count,
                'correct': self.correct_count,
                'failed': self.failed_count,
                'total': self.completed_count + self.failed_count
            }


class ProgressManager:
    """Progress manager"""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.lock = threading.Lock()
        self.completed = 0
        self.correct = 0
        self.failed = 0
        self.progress_bar = None
        self._setup_progress_bar()
    
    def _setup_progress_bar(self):
        """Setup progress bar"""
        self.progress_bar = tqdm(
            total=self.total_tasks, 
            desc="Processing progress",
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
        )
        self._update_display()
    
    def update_result(self, is_completed: bool, is_correct: bool = False, is_failed: bool = False):
        """Update processing result"""
        with self.lock:
            if is_failed:
                self.failed += 1
            elif is_completed:
                self.completed += 1
                if is_correct:
                    self.correct += 1
            
            self.progress_bar.update(1)
            self._update_display()
    
    def _calculate_accuracy(self) -> float:
        """Calculate accuracy"""
        return self.correct / self.completed if self.completed > 0 else 0.0
    
    def _update_display(self):
        """Update progress bar display"""
        if not self.progress_bar:
            return
            
        accuracy = self._calculate_accuracy()
        
        if self.failed > 0:
            postfix = f"Accuracy: {accuracy:.1%}({self.correct}/{self.completed}) [Failed:{self.failed}]"
        else:
            postfix = f"Accuracy: {accuracy:.1%}({self.correct}/{self.completed})"
        
        self.progress_bar.set_postfix_str(postfix)
    
    def close(self):
        """Close progress bar"""
        if self.progress_bar:
            self.progress_bar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ThreadLocalGeminiClient:
    """Thread-local Gemini client manager"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._local = threading.local()
    
    def get_client(self):
        """Get client for current thread"""
        if not hasattr(self._local, 'client'):
            self._local.client = genai.Client(api_key=self.api_key)
        return self._local.client


class EvaluationConfig:
    """Configuration class"""
    
    def __init__(self, 
                 api_key: str,
                 model: str = "gemini-2.5-flash",
                 data_json_file: str = "./merged_qas_1_0817.json",
                 video_dir: str = "./video/",
                 max_workers: int = 1,
                 no_sound: bool = False):
        self.api_key = api_key
        self.model = model
        self.data_json_file = data_json_file
        self.video_dir = video_dir
        self.max_workers = max_workers
        self.no_sound = no_sound


class TextProcessor:
    """Text processing utility class"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean model answer"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        translator = str.maketrans('', '', string.punctuation)
        text_clean = text.translate(translator)
        text_clean = ' '.join(text_clean.split())
        return text_clean
    
    @staticmethod
    def clean_json_response(response_text: str) -> str:
        """Clean response text"""
        if response_text.startswith("```json\n"):
            response_text = response_text[8:]
        if response_text.endswith("\n```"):
            response_text = response_text[:-4]
        return response_text


class GeminiEvaluator:
    """Gemini model evaluator"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.dataloader = VideoQADaloader(
            data_json_file=config.data_json_file, 
            video_dir=config.video_dir
        )
        self.text_processor = TextProcessor()
        self.client_manager = ThreadLocalGeminiClient(config.api_key)
        self.result_collector = ThreadSafeResultCollector()
        self.file_write_lock = threading.Lock()
        
        if config.no_sound:
            temp_dir = os.path.join(os.getcwd(), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            print(f"Temp directory created: {temp_dir}")
    
    def _check_video_exists(self, video_path: str) -> bool:
        """Check if video file exists"""
        return os.path.exists(video_path) and os.path.isfile(video_path)
    
    def _remove_audio_from_video_thread_safe(self, video_path: str) -> str:
        """Thread-safe removal of audio from video"""
        if not self.config.no_sound:
            return video_path
        
        try:
            temp_dir = os.path.join(os.getcwd(), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            video_name = os.path.basename(video_path)
            name, ext = os.path.splitext(video_name)
            thread_id = threading.current_thread().ident
            timestamp = int(time.time() * 1000)
            temp_filename = f"{name}_no_sound_{thread_id}_{timestamp}{ext}"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            print(f"[Thread {thread_id}] Removing audio: {video_path} -> {temp_path}")
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-c:v', 'copy',
                '-an',
                '-y',
                '-loglevel', 'quiet',
                temp_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                check=True
            )
            
            if not os.path.exists(temp_path):
                raise Exception("Failed to create temp file without audio")
            
            print(f"[Thread {thread_id}] Audio removal complete: {temp_path}")
            return temp_path
            
        except Exception as e:
            thread_id = threading.current_thread().ident
            print(f"[Thread {thread_id}] Failed to remove audio: {e}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise
    
    def _cleanup_temp_files(self):
        """Clean up all temporary files"""
        if not self.config.no_sound:
            return
            
        print("Starting cleanup of temporary audio-removed files...")
        
        try:
            temp_dir = os.path.join(os.getcwd(), "temp")
            if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"Deleted temp file: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete temp file: {file_path}, Error: {e}")
                
                if not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                    print("Deleted empty temp folder")
        except Exception as e:
            print(f"Error while cleaning temp folder: {e}")
        
        print("Temp file cleanup complete")
    
    def _build_prompt(self, question: str, options: List[str]) -> str:
        """Build prompt"""
        options_text = "\n".join(options)
        prompt = (
            "You are given a video. Based on the content of the video, answer the following question:\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{options_text}\n\n"
            "If your access to the video content is limited, at least one option that is more likely than the others must be chosen."
            "Directly return a JSON structure without any additional explanations: { \"answer\": \"A/B/C/D\", \"explanation\": \"string\" // the explanation you give }"
        )
        return prompt
    
    def _upload_file_thread_safe(self, video_path: str, client):
        """Thread-safe file upload"""
        try:
            thread_id = threading.current_thread().ident
            processed_video_path = self._remove_audio_from_video_thread_safe(video_path)
            suffix = " (no audio)" if self.config.no_sound else ""

            try:
                myfile = client.files.upload(file=processed_video_path)
            except Exception as e:
                # Check if it's a 500 error (based on error string content and error code)
                error_str = str(e)
                is_500 = False
                # 1. Check string content
                if '500 INTERNAL' in error_str:
                    is_500 = True
                # 2. Check error code
                if hasattr(e, 'args') and len(e.args) > 0:
                    arg0 = e.args[0]
                    if isinstance(arg0, dict):
                        if 'error' in arg0 and isinstance(arg0['error'], dict):
                            if arg0['error'].get('code', None) == 500:
                                is_500 = True
                # 3. Check response.status_code
                if hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 500:
                    is_500 = True

                if is_500:
                    print(f"[Thread {thread_id}] File upload failed (500): {video_path}{suffix}, skipping this file. Error: {e}")
                    # Clean up temp file
                    if processed_video_path != video_path and os.path.exists(processed_video_path):
                        try:
                            os.remove(processed_video_path)
                        except:
                            pass
                    # Return None to indicate skip
                    return None
                # Re-raise other exceptions
                raise

            retry_count = 0
            max_retries = 100
            while myfile.state.name == 'PROCESSING' and retry_count < max_retries:
                time.sleep(10)
                try:
                    myfile = client.files.get(name=myfile.name)
                    retry_count += 1
                except Exception as e:
                    print(f"[Thread {thread_id}] Failed to check file status: {e}")
                    time.sleep(60)
                    retry_count += 1

            if myfile.state.name == 'FAILED':
                raise Exception("File processing failed")
            elif myfile.state.name == 'PROCESSING':
                raise Exception("File processing timeout")

            if processed_video_path != video_path and os.path.exists(processed_video_path):
                try:
                    os.remove(processed_video_path)
                    print(f'[Thread {thread_id}] Cleaned up temp file: {processed_video_path}')
                except Exception as e:
                    print(f"[Thread {thread_id}] Failed to clean up temp file: {e}")

            return myfile

        except Exception as e:
            thread_id = threading.current_thread().ident
            print(f"[Thread {thread_id}] File upload failed: {video_path}{suffix}, Error: {e}")

            if 'processed_video_path' in locals() and processed_video_path != video_path and os.path.exists(processed_video_path):
                try:
                    os.remove(processed_video_path)
                except:
                    pass
            raise

    def _get_model_response_thread_safe(self, video_path: str, prompt: str, client) -> Dict[str, Any]:
        """Thread-safe model response retrieval"""
        thread_id = threading.current_thread().ident
        uploaded_file = None
        
        try:
            uploaded_file = self._upload_file_thread_safe(video_path, client)
            
            response = client.models.generate_content(
                model=self.config.model,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0
                )
            )
            
            try:
                response_data = json.loads(self.text_processor.clean_json_response(response.text))
            except Exception as e:
                match = re.search(r'\\boxed\{(?:\\text\{)?([A-D])\}?\}', response.text)
                if match:
                    answer = match.group(1)
                    explanation = re.sub(r'\$?\\boxed\{(?:\\text\{)?[A-D]\}?\}\$?', '', response.text).strip()
                    response_data = {"answer": answer, "explanation": explanation}
                else:
                    letter_match = re.search(r'\b([A-D])\b', response.text)
                    if letter_match:
                        answer = letter_match.group(1)
                        explanation = response.text.strip()
                        response_data = {"answer": answer, "explanation": explanation}
                    else:
                        raise Exception(f"Unable to parse model response: {response.text}")

            return response_data
            
        except Exception as e:
            print(f"[Thread {thread_id}] API call failed: {e}")
            raise
        
        finally:
            if uploaded_file:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception as e:
                    print(f"[Thread {thread_id}] Failed to clean up uploaded file: {e}")
    
    def _evaluate_single_qa_thread_safe(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """Thread-safe evaluation of single QA pair"""
        thread_id = threading.current_thread().ident
        video_path = qa_pair.get("video_path", "")
        question = qa_pair.get("question")
        options = qa_pair.get("options")
        answer = qa_pair.get("answer")
        
        result = {
            'video_path': video_path,
            'question': question,
            'options': options,
            'answer': answer,
            'model_answer': None,
            'is_correct': False,
            'error_message': None
        }
        
        if not all([video_path, question, options, answer]):
            error_msg = f"Incomplete data: video_path={video_path}, question={bool(question)}, options={bool(options)}, answer={bool(answer)}"
            print(f"[Thread {thread_id}] Skipping incomplete item: {error_msg}")
            result['model_answer'] = 'Incomplete data'
            result['error_message'] = error_msg
            return result
        
        if not self._check_video_exists(video_path):
            error_msg = f"Video file does not exist: {video_path}"
            print(f"[Thread {thread_id}] {error_msg}")
            result['model_answer'] = 'Video file does not exist'
            result['error_message'] = error_msg
            return result
        
        try:
            client = self.client_manager.get_client()
            prompt = self._build_prompt(question, options)
            response = self._get_model_response_thread_safe(video_path, prompt, client)
                
            model_answer = response['answer']
            model_explanation = response['explanation']
                
            result['model_answer'] = model_answer
            result['model_explanation'] = model_explanation
            
            cleaned_model_answer = self.text_processor.clean_text(model_answer)
            cleaned_answer = self.text_processor.clean_text(answer)
            
            result['is_correct'] = (cleaned_model_answer == cleaned_answer)
                
        except Exception as e:
            error_msg = f"Error processing video {video_path}: {str(e)}"
            print(f"[Thread {thread_id}] {error_msg}")
            print(f"[Thread {thread_id}] Exception details: {traceback.format_exc()}")
            result['model_answer'] = f"Error: {str(e)}"
            result['error_message'] = error_msg
        
        return result
    
    def process_single_qa_independently(self, qa_pair: Dict[str, Any], qa_index: int, 
                                      thread_id: str, progress_manager: ProgressManager = None):
        """Process single QA task independently"""
        try:
            result = self._evaluate_single_qa_thread_safe(qa_pair)
            self.result_collector.add_result(result)
            
            if progress_manager:
                if result.get('error_message'):
                    progress_manager.update_result(is_completed=False, is_failed=True)
                    status = 'failed'
                else:
                    progress_manager.update_result(is_completed=True, is_correct=result['is_correct'])
                    status = 'completed'
            else:
                status = 'failed' if result.get('error_message') else 'completed'
            
            return status
                
        except Exception as e:
            error_msg = f"QA task {qa_pair.get('video_path', 'unknown')} processing exception: {str(e)}"
            print(f"\n[Thread {thread_id}] {error_msg}")
            
            error_result = {
                'video_path': qa_pair.get("video_path", ""),
                'question': qa_pair.get("question", ""),
                'options': qa_pair.get("options", []),
                'answer': qa_pair.get("answer", ""),
                'model_answer': f"Thread exception: {str(e)}",
                'is_correct': False,
                'error_message': error_msg
            }
            self.result_collector.add_result(error_result)
            
            if progress_manager:
                progress_manager.update_result(is_completed=False, is_failed=True)
            
            return 'failed'
    
    def _save_results_to_file(self, output_file: str):
        """Save results to file (without .bak file)"""
        try:
            all_results = self.result_collector.get_all_results()
            
            if not all_results:
                print(f"No results to save to {output_file}")
                return
            
            with self.file_write_lock:
                results_to_save = []
                for result in all_results:
                    complete_result = {
                        'video_path': result.get('video_path', ''),
                        'question': result.get('question', ''),
                        'options': result.get('options', []),
                        'answer': result.get('answer', ''),
                        'model_answer': result.get('model_answer', ''),
                        'is_correct': result.get('is_correct', False),
                        'error_message': result.get('error_message', None)
                    }
                    if 'model_explanation' in result:
                        complete_result['model_explanation'] = result['model_explanation']
                    results_to_save.append(complete_result)
                
                temp_file = output_file + ".tmp"
                try:
                    # Write to temp file first
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(results_to_save, f, indent=4, ensure_ascii=False)
                    
                    # Directly overwrite original file without creating .bak file
                    if os.path.exists(temp_file):
                        # Windows requires removing target file first
                        if os.path.exists(output_file):
                            os.remove(output_file)
                        os.rename(temp_file, output_file)
                    
                except Exception as write_error:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    raise write_error
                
        except Exception as e:
            print(f"Failed to save results to file: {e}")
            print(f"Detailed error: {traceback.format_exc()}")
            raise
    
    def _load_existing_results(self, output_file: str) -> Dict[str, Any]:
        """Load existing results file"""
        if not os.path.exists(output_file):
            return {"results": [], "correct_count": 0}
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                results = data
                # Filter out results with non-null error_message
                valid_results = [r for r in results if r.get('error_message') is None]
                error_results = [r for r in results if r.get('error_message') is not None]
                
                correct_count = sum(1 for result in valid_results if result.get('is_correct', False))
                
                model_name = os.path.splitext(os.path.basename(output_file))[0].replace('_qsa_2', '')
                suffix = " (no audio)" if self.config.no_sound else ""
                print(f"Loaded {model_name}{suffix}: {len(valid_results)} valid, {len(error_results)} to retry, {correct_count} correct")
                
                # Return only valid results
                return {
                    "results": valid_results,
                    "correct_count": correct_count
                }
            else:
                results = data.get('results', [])
                # Filter out results with non-null error_message
                valid_results = [r for r in results if r.get('error_message') is None]
                error_results = [r for r in results if r.get('error_message') is not None]
                
                correct_count = sum(1 for result in valid_results if result.get('is_correct', False))
                
                model_name = os.path.splitext(os.path.basename(output_file))[0].replace('_qsa_2', '')
                suffix = " (no audio)" if self.config.no_sound else ""
                print(f"Loaded {model_name}{suffix}: {len(valid_results)} valid, {len(error_results)} to retry, {correct_count} correct")
            
                return {
                    "results": valid_results,
                    "correct_count": correct_count
                }
        except Exception as e:
            print(f"Failed to load existing results file: {e}")
            return {"results": [], "correct_count": 0}

    def evaluate_single_model_multithreaded(self, output_file: str) -> Dict[str, Any]:
        """Run multi-threaded evaluation for single model"""
        print("Starting true multi-threaded evaluation mode - lock-free concurrent processing")
        
        all_qa_pairs = self.dataloader.get_all_qa_pairs()
        
        if self.config.no_sound and not output_file.endswith('_no_sound.json'):
            base_name = os.path.splitext(output_file)[0]
            output_file = f"{base_name}_no_sound.json"
        
        existing_data = self._load_existing_results(output_file)
        existing_results = existing_data["results"]
        
        # Create set of processed items (only results without errors)
        processed_items = set()
        for result in existing_results:
            video_path = result.get('video_path', '')
            question = result.get('question', '')
            item_key = f"{video_path}|||{question}"
            processed_items.add(item_key)
        
        # Filter out unprocessed QA pairs
        remaining_qa_pairs = []
        for i, qa_pair in enumerate(all_qa_pairs):
            video_path = qa_pair.get("video_path", "")
            question = qa_pair.get("question", "")
            item_key = f"{video_path}|||{question}"
            if item_key not in processed_items:
                remaining_qa_pairs.append((qa_pair, i))
        
        total_pairs = len(all_qa_pairs)
        remaining_count = len(remaining_qa_pairs)
        
        if remaining_count == 0:
            print(f"All QA pairs for model {self.config.model} have been processed!")
            return {
                "total": len(existing_results),
                "correct": existing_data["correct_count"],
                "accuracy": existing_data["correct_count"] / len(existing_results) if existing_results else 0.0
            }
        
        print(f"Evaluation task statistics:")
        print(f"   Total QA pairs: {total_pairs}")
        print(f"   Completed: {len(existing_results)}")
        print(f"   Remaining: {remaining_count}")
        print(f"   Thread pool size: {self.config.max_workers}")
        print(f"   Processing mode: {'No audio mode' if self.config.no_sound else 'Full audio mode'}")
        print(f"   Output file: {output_file}")
        
        try:
            # Add existing results to result collector
            for result in existing_results:
                self.result_collector.add_result(result)
            
            # Execute multi-threaded evaluation
            final_stats = self._execute_true_multithreaded_evaluation(remaining_qa_pairs, output_file)
            
            print(f"Multi-threaded processing complete!")
            return final_stats
            
        except KeyboardInterrupt:
            print("\nUser interrupted evaluation process, saving safely...")
            self._save_results_to_file(output_file)
            print(f"Progress saved to: {output_file}")
            raise
        except Exception as e:
            print(f"\nError during evaluation: {e}")
            self._save_results_to_file(output_file)
            print(f"Progress saved to: {output_file}")
            raise
    
    def _execute_true_multithreaded_evaluation(self, remaining_qa_pairs: List[Tuple], output_file: str) -> Dict[str, Any]:
        """Execute multi-threaded evaluation"""
        total_tasks = len(remaining_qa_pairs)
        
        with ProgressManager(total_tasks) as progress_manager:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                print(f"Starting {self.config.max_workers} concurrent threads, beginning lock-free processing...")
                
                futures = []
                future_to_qa = {}
                
                for i, (qa_pair, qa_index) in enumerate(remaining_qa_pairs):
                    thread_id = f"T{i%self.config.max_workers+1}"
                    try:
                        future = executor.submit(
                            self.process_single_qa_independently,
                            qa_pair, qa_index, thread_id, progress_manager
                        )
                        futures.append(future)
                        future_to_qa[future] = (qa_pair.get('video', 'unknown'), thread_id)
                    except Exception as e:
                        print(f"Failed to submit task: {e}")
                
                print(f"Successfully submitted {len(futures)} concurrent tasks")
                
                completed_count = 0
                failed_count = 0
                start_time = time.time()
                
                last_save_time = start_time
                save_interval = 30
                
                for future in as_completed(futures):
                    try:
                        video_name, thread_id = future_to_qa.get(future, ('unknown', 'unknown'))
                        status = future.result(timeout=300)
                        completed_count += 1
                        
                        if status == 'failed':
                            failed_count += 1
                        
                        current_time = time.time()
                        if current_time - last_save_time >= save_interval:
                            self._save_results_to_file(output_file)
                            last_save_time = current_time
                        
                    except Exception as e:
                        video_name, thread_id = future_to_qa.get(future, ('unknown', 'unknown'))
                        print(f"Task execution exception {video_name} (Thread:{thread_id}): {e}")
                        failed_count += 1
                        completed_count += 1
                
                # Final save
                self._save_results_to_file(output_file)
                
                # Get final statistics
                final_stats = self.result_collector.get_stats()
                total_time = time.time() - start_time
                
                print(f"\nMulti-threaded evaluation complete!")
                print(f"Final statistics:")
                print(f"   Total processing time: {total_time/60:.1f} minutes")
                print(f"   Average processing speed: {final_stats['total']/total_time:.1f} tasks/sec")
                print(f"   Total processed: {final_stats['total']}")
                print(f"   Successful: {final_stats['completed']}")
                print(f"   Correct: {final_stats['correct']}")
                print(f"   Failed: {final_stats['failed']}")
                print(f"   Accuracy: {final_stats['correct']/final_stats['completed']*100:.1f}%" if final_stats['completed'] > 0 else "   Accuracy: 0%")
                
                return {
                    "total": final_stats['total'],
                    "correct": final_stats['correct'],
                    "accuracy": final_stats['correct'] / final_stats['completed'] if final_stats['completed'] > 0 else 0.0
                }


    def evaluate_single_model_singlethreaded(self, output_file: str) -> Dict[str, Any]:
        """Run single-threaded evaluation for single model"""
        print("Starting single-threaded evaluation mode")
        all_qa_pairs = self.dataloader.get_all_qa_pairs()
        
        if self.config.no_sound and not output_file.endswith('_no_sound.json'):
            base_name = os.path.splitext(output_file)[0]
            output_file = f"{base_name}_no_sound.json"
        
        existing_data = self._load_existing_results(output_file)
        existing_results = existing_data["results"]
        
        processed_items = set()
        for result in existing_results:
            video = result.get('video', '')
            question = result.get('question', '')
            item_key = f"{video}|||{question}"
            processed_items.add(item_key)
        
        remaining_qa_pairs = []
        for i, qa_pair in enumerate(all_qa_pairs):
            video_name = qa_pair.get("video", "")
            question = qa_pair.get("question", "")
            item_key = f"{video_name}|||{question}"
            if item_key not in processed_items:
                remaining_qa_pairs.append((qa_pair, i))
        
        total_pairs = len(all_qa_pairs)
        remaining_count = len(remaining_qa_pairs)
        
        if remaining_count == 0:
            print(f"All QA pairs for model {self.config.model} have been processed!")
            return {
                "total": len(existing_results),
                "correct": existing_data["correct_count"],
                "accuracy": existing_data["correct_count"] / len(existing_results) if existing_results else 0.0
            }
        
        print(f"Evaluation task statistics:")
        print(f"   Total QA pairs: {total_pairs}")
        print(f"   Completed: {len(existing_results)}")
        print(f"   Remaining: {remaining_count}")
        print(f"   Processing mode: {'No audio mode' if self.config.no_sound else 'Full audio mode'}")
        print(f"   Output file: {output_file}")
        
        try:
            for result in existing_results:
                self.result_collector.add_result(result)
            
            total_tasks = len(remaining_qa_pairs)
            start_time = time.time()
            last_save_time = start_time
            save_interval = 30
            
            with ProgressManager(total_tasks) as progress_manager:
                for i, (qa_pair, qa_index) in enumerate(remaining_qa_pairs):
                    _ = self.process_single_qa_independently(qa_pair, qa_index, thread_id="S1", progress_manager=progress_manager)
                    current_time = time.time()
                    if current_time - last_save_time >= save_interval:
                        self._save_results_to_file(output_file)
                        last_save_time = current_time
            
            self._save_results_to_file(output_file)
            final_stats = self.result_collector.get_stats()
            total_time = time.time() - start_time
            
            print(f"\nSingle-threaded evaluation complete!")
            print(f"Final statistics:")
            print(f"   Total processing time: {total_time/60:.1f} minutes")
            print(f"   Average processing speed: {final_stats['total']/total_time:.1f} tasks/sec")
            print(f"   Total processed: {final_stats['total']}")
            print(f"   Successful: {final_stats['completed']}")
            print(f"   Correct: {final_stats['correct']}")
            print(f"   Failed: {final_stats['failed']}")
            print(f"   Accuracy: {final_stats['correct']/final_stats['completed']*100:.1f}%" if final_stats['completed'] > 0 else "   Accuracy: 0%")
            
            return {
                "total": final_stats['total'],
                "correct": final_stats['correct'],
                "accuracy": final_stats['correct'] / final_stats['completed'] if final_stats['completed'] > 0 else 0.0
            }
        except KeyboardInterrupt:
            print("\nUser interrupted evaluation process, saving safely...")
            self._save_results_to_file(output_file)
            print(f"Progress saved to: {output_file}")
            raise
        except Exception as e:
            print(f"\nError during evaluation: {e}")
            self._save_results_to_file(output_file)
            print(f"Progress saved to: {output_file}")
            raise

def main_with_args(models: List[str] = None,
                   max_workers: int = 15,
                   no_sound: bool = False,
                   multithread: bool = False,
                   input_file: str = 'qa_data.json',
                   video_dir: str = './video/',
                   api_key: str = None):  
    """Main function"""
    if models is None:
        models = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
    
    # check API key
    if not api_key:
        raise ValueError("API key is required. Please provide it using --api_key argument.")
    
    print("=" * 80)
    print("Gemini Evaluator")
    if multithread:
        print("Mode: Multi-threaded concurrent processing")
        print(f"Concurrent threads: {max_workers}")
    else:
        print("Mode: Single-threaded sequential processing")
    if no_sound:
        print("No audio mode enabled")
    print("=" * 80)
    print(f"Models to evaluate: {', '.join(models)}")
    print(f"Input file: {input_file}")
    print(f"Video directory: {video_dir}")
    print("=" * 80)
    
    try:
        # Process each model one by one
        for model in models:
            print(f"\nStarting evaluation for model: {model}")
            
            config = EvaluationConfig(
                api_key=api_key, 
                model=model,
                data_json_file=input_file,
                video_dir=video_dir,
                max_workers=max_workers,
                no_sound=no_sound
            )
            
            evaluator = GeminiEvaluator(config)
            
            input_file_base = os.path.splitext(input_file)[0]
            if no_sound:
                output_file = f"{model}_{input_file_base}_no_sound.json"
            else:
                output_file = f"{model}_{input_file_base}-2.json"
            
            print(f"Output file: {output_file}")
            
            if multithread:
                result = evaluator.evaluate_single_model_multithreaded(output_file)
            else:
                result = evaluator.evaluate_single_model_singlethreaded(output_file)
            
            print(f"\nModel {model} evaluation complete!")
            print(f"   Accuracy: {result['accuracy']:.2%}")
            print(f"   Correct: {result['correct']}/{result['total']}")
            
            # Clean up temp files
            evaluator._cleanup_temp_files()
        
        print("\n" + "=" * 80)
        print("All model evaluations complete!")
        
        # Display summary results
        print("\nSummary results:")
        for model in models:
            input_file_base = os.path.splitext(input_file)[0]
            if no_sound:
                output_file = f"{model}_{input_file_base}_no_sound.json"
            else:
                output_file = f"{model}_{input_file_base}.json"
            
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    total = len(data)
                    correct = sum(1 for item in data if item.get('is_correct', False) and not item.get('error_message'))
                    accuracy = correct / total if total > 0 else 0.0
                    
                    print(f"   {model}: {accuracy:.2%} ({correct}/{total}) -> {output_file}")
                except Exception as e:
                    print(f"   {model}: Failed to read results - {e}")
        
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nUser interrupted evaluation process")
        print("Progress saved, you can rerun the program to continue evaluation")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print(f"Detailed error: {traceback.format_exc()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gemini Multi-threaded Evaluator')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro'],
                       help='List of models to evaluate')
    parser.add_argument('-w', '--max_workers', type=int, default=15,
                       help='Maximum concurrent threads (default: 15)')
    parser.add_argument('--no_sound', action='store_true',
                       help='Enable no audio mode')
    parser.add_argument('--multithread', action='store_true',
                       help='Enable multi-threaded mode (default: off, single-threaded)')
    parser.add_argument('--input_file', type=str, default='data.json',
                       help='Path to the input JSON file containing QA pairs (default: qa_data.json)')
    parser.add_argument('--video_dir', type=str, default='./videos',
                       help='Directory where video files are located (default: ./video/)')
    parser.add_argument('--api_key', type=str, required=True,
                       help='Your Google AI API key')
    
    args = parser.parse_args()

    main_with_args(models=args.models, max_workers=args.max_workers, no_sound=args.no_sound, multithread=args.multithread, input_file=args.input_file, video_dir=args.video_dir, api_key=args.api_key)
