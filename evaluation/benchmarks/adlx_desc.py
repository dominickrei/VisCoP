import json
import os
import re
from typing import Dict, Any, Union

import pandas as pd

from .base import BaseVideoEvalDataset

import videochatgpt_scoring

class ADLX_Descriptions(BaseVideoEvalDataset):

    BENCHMARK_TYPE: str = "oqa"

    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}
        idx = 0

        json_file = os.path.join(data_root, 'Charades-Description.json')
        video_folder = os.path.join(data_root, 'videos/Charades_v1_480')

        samples = json.load(open(json_file))
        
        for sample_id, sample in enumerate(samples):
            video_filename = sample['video_name'] + '.mp4'

            qa_pairs = [
                ('general', sample['Q'], sample['A']),
                ('cons1', sample['cons_Q1'], sample['cons_A']),
                ('cons2', sample['cons_Q2'], sample['cons_A'])
            ]

            for qa_type, question, answer in qa_pairs:
                data_dict[idx] = {
                    # required fields for data loading
                    "video_path": os.path.join(video_folder, str(video_filename)),
                    "start_time": None,
                    "end_time": None,
                    # required fields for evaluation
                    "task_type": qa_type,
                    "ground_truth": answer,
                    # custom fields for instruction generation and post processing
                    "question": question,
                    "sample_id": sample_id
                }
                idx += 1

        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video: Any) -> str:
        meta_data = self.data_dict[data_id]
        instruction = meta_data["question"]

        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> int:
        return response

    def evaluate(self, results):
        aggregated_results = {}

        ### First build a result dict that is easy to parse, contains general and consistency Q/A
        for sample in results:
            meta_data = self.data_dict[sample['data_id']]
            sample_id = meta_data['sample_id']

            # initial creation of dict entry
            if not sample_id in aggregated_results:
                aggregated_results[sample_id] = {
                    'video_path': meta_data['video_path']
                }

            sample_type = meta_data['task_type']
            aggregated_results[sample_id][sample_type + '_Q'] = meta_data['question']
            aggregated_results[sample_id][sample_type + '_predA'] = sample['prediction']

            if 'cons' in sample_type:
                aggregated_results[sample_id]['cons_gtA'] = meta_data['ground_truth']
            else:
                aggregated_results[sample_id]['general_gtA'] = meta_data['ground_truth']

        ### Compute the scores for each sample
        for sample_id, data in aggregated_results.items():
            # compute general scores
            general_question = data['general_Q']
            general_pred = data['general_predA']
            general_gt = data['general_gtA']
            
            correctness = videochatgpt_scoring.get_correctness_score(general_question, general_gt, general_pred)
            detail_orientation = videochatgpt_scoring.get_detail_orientation_score(general_question, general_gt, general_pred)
            contextual = videochatgpt_scoring.get_context_score(general_question, general_gt, general_pred)
            temporal = videochatgpt_scoring.get_temporal_score(general_question, general_gt, general_pred)

            # compute consistency scores
            cons1_Q = data['cons1_Q']
            cons2_Q = data['cons2_Q']
            cons1_pred = data['cons1_predA']
            cons2_pred = data['cons2_predA']
            cons_gt = data['cons_gtA']

            consistency = videochatgpt_scoring.get_consistency_score(cons1_Q, cons2_Q, cons_gt, cons1_pred, cons2_pred)

            data['metrics'] = {
                'correctness': correctness,
                'detail_orientation': detail_orientation,
                'contextual': contextual,
                'temporal': temporal,
                'consistency': consistency
            }
        
        ### Convert to more efficient save format, compute aggregate metrics
        summed_metrics = [0, 0, 0, 0, 0]
        save_results = []

        # summary metrics
        save_results.append(
            {
                'correctness': 0,
                'detail_orientation': 0,
                'contextual': 0,
                'temporal': 0,
                'consistency': 0,
                'average': 0
            }
        )

        # sum all metrics, append data to list we will save
        for sample_id, data in aggregated_results.items():
            data['sample_id'] = sample_id
            for i, k in enumerate(['correctness', 'detail_orientation', 'contextual', 'temporal', 'consistency']):
                summed_metrics[i] += data['metrics'][k]
            
            save_results.append(data)

        # compute summary statistics
        for i, k in enumerate(['correctness', 'detail_orientation', 'contextual', 'temporal', 'consistency']):
            averaged_metric = summed_metrics[i] / len(aggregated_results)
            save_results[0][k] = averaged_metric * 20 # x20 to scale. done in initial ADLX paper

        save_results[0]['average'] = sum(list(save_results[0].values())[:-1]) / 5 # 5 is number of tasks

        return save_results[0], save_results