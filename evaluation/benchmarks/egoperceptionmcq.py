import json
import os
import re
from typing import Dict, Any, Union

import pandas as pd

from .base import BaseVideoEvalDataset

import llm_answer_parsing

class EgoExo4DDataset(BaseVideoEvalDataset):

    BENCHMARK_TYPE: str = "mcqa"

    def load_data(self, data_root: str, eval_views: list[int] = ['ego', 'exo'], video_dir: str = "egocentric_videos") -> Dict[int, Any]:
        data_dict = {}
        idx = 0

        json_file = os.path.join(data_root, "all_category_qas.json")
        video_folder = os.path.join(data_root, video_dir)

        samples = json.load(open(json_file))

        types = []
        for sample in samples:
            answer = sample['ground_truth_letter']
            options = [v for k, v in sample['answer_choices'].items()]
            option_letters = [k for k, v in sample['answer_choices'].items()]
            category_str = sample['category_name']
            question = sample['question']

            for view in eval_views:
                video_filename = sample['video_filename']
                if view == 'ego':
                    video_filename = video_filename[:-4] + 'ego'

                data_dict[idx] = {
                    # required fields for data loading
                    "video_path": os.path.join(video_folder, str(video_filename) + '.mp4'),
                    "start_time": None,
                    "end_time": None,
                    # required fields for evaluation
                    "task_type": [view, category_str + f'_{view}'],
                    "ground_truth": option_letters.index(answer),
                    # custom fields for instruction generation and post processing
                    "question": question,
                    "options": options,
                    "option_letters": option_letters,
                }
                idx += 1

        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video: Any) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        option_letters = meta_data["option_letters"]
        options = meta_data["options"]

        choices_str = " ".join(f'({letter}) {option}' for letter, option in zip(option_letters, options))
        instruction = f"{question} The output should be the choice among one of the following choices. Choices are {choices_str}"

        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> int:
        meta_data = self.data_dict[data_id]
        options = meta_data["options"]
        option_letters = meta_data["option_letters"]

        response = response.replace('answer', '')
        response = response.replace('Answer', '')
        max_letter = sorted(option_letters)[-1]
        pred_answer = re.findall(f'[\(\ ]*[A-{max_letter}][\)\ ]*', response)

        find_flag = False
        # regex cant parse model response. see if llama 3.1 can recover the correct answer using our models outputs
        if len(pred_answer) == 0:
            question = meta_data['question']
            choices_str = " ".join(f'({letter}) {option}' for letter, option in zip(option_letters, options))
            prompt = llm_answer_parsing.build_prompt(question, choices_str, response)
            llm_response = llm_answer_parsing.parse_with_llama(prompt)

            pred_answer = re.findall(f'[\(\ ]*[A-{max_letter}][\)\ ]*', response)
        
            if len(pred_answer) == 0: # still cant parse after llm call
                for idx, opt in enumerate(options):
                    opt = opt.strip()
                    opt = opt.strip('.')
                    if opt.lower() in response.lower():
                        pred_idx = idx
                        find_flag = True
                        break

        # regex could parse the answer without using llama 3.1
        if not len(pred_answer) == 0:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = option_letters.index(pred_answer)
            find_flag = True

        if not find_flag:
            pred_idx = -1 # count this as a failure, model output couldnt be parsed

        # assert find_flag, f"Cannot find the answer in the options: {response}"
        return pred_idx

# class EgoExo4DDataset_Ego(EgoExo4DDataset):
#     def load_data(self, data_root: str) -> Dict[int, Any]:
#         data_dict = super().load_data(data_root)
#         return data_dict

class EgoExo4DDataset_Depth(EgoExo4DDataset):
    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = super().load_data(data_root, eval_views=['exo'], video_dir="depth_videos")
        return data_dict