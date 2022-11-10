from typing import Dict, List
import json
from gluonts.dataset.common import ListDataset
import subprocess
from gluonts.dataset.field_names import FieldName
from typing import Callable, Iterable, Iterator, List
import torch
from pts.model.deepar import DeepAREstimator
from pts import Trainer
import os
from pathlib import Path
from gluonts.torch.model.predictor import PyTorchPredictor
import numpy as np
import datetime

class AlgSolution():
    def __init__(self):
        pass

    def train_model(self, input_data_path: str, output_model_path: str, params: Dict, **kwargs) -> bool:
        """
        :param input_data_path: 本地输入数据集路径
        :param output_model_path: 本地输出模型路径
        :param params: 训练输入参数, 默认为conf/default.json
        :param kwargs:
        :return: bool: True 成功; False 失败
        """

        train_data_path = os.path.join(input_data_path, 'train.jsonl')
        train_samples = []
        with open(train_data_path) as f:
            for line in f:
                sample = json.loads(line)
                train_samples.append(sample)


        # Args:
        # input_data_path (str): 本地输入数据集路径
        # output_model_path (str): 本地输出模型路径
        # params (Dict): 训练输入参数。默认为conf/default.json
        # Returns:
        # bool: True 成功; False 失败

