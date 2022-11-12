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
# import torch.distributed as dist
import numpy as np
import datetime
from utils.json_utils import sample_train_validate, construct_cheat_dict
from sklearn.preprocessing import LabelEncoder
from pts.modules import StudentTOutput, StudentTMixtureOutput
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class AlgSolution():
    def __init__(self):
        pass

    def train_model(self, input_data_path: str, output_model_path: str, params: Dict, **kwargs) -> bool:

        """
        :param input_data_path: 输入路径名称
        :param output_model_path: 输出模型地址
        :param params: 参数结果
        :param kwargs: 其他别的
        :return: bool: True 成功; False 失败
        """


        train_data_path = os.path.join(input_data_path, 'train.jsonl')

        print(device)
        train_samples = []

        # 训练样本、验证样本的list
        train_sample_list, valid_sample_list = list(), list()
        # 全部样本集合
        sample_list = list()


        # cate类列的名字[选择哪些]
        cate_column = ["item_id", "app_id", "zone_id"][0:2]

        # 计算embedding所需要的内容大小
        cate_column_set = [list() for _ in range(len(cate_column))]

        # embedding参数作弊表
        cate_column_dict = dict(zip(cate_column, cate_column_set))


        # 归一化作弊表
        cate_scaler_list = [LabelEncoder() for _ in range(len(cate_column))]
        cate_scaler_dict = dict(zip(cate_column, cate_scaler_list))
        cate_column_count_dict = dict(zip(cate_column, cate_scaler_list))


        with open(train_data_path) as f:
            for line in f:
                # 读取sample结果
                sample = json.loads(line)
                sample_list.append(sample)

                train_sample, valid_sample = sample_train_validate(sample=sample, validation_length=144 * 2, box_plot=True)
                train_sample_list.append(train_sample)
                valid_sample_list.append(valid_sample)

                # 添加embedding参数
                for cate_id in cate_column:
                    cate_column_dict[cate_id].append(sample[cate_id])


                sample = json.loads(line)
                train_samples.append(sample)

        # 写作弊表信息
        for cate_id in cate_column:
            # 标准化数量
            # cate_scaler_dict[cate_id] = cate_scaler_dict[cate_id].fit(cate_column_dict[cate_id])
            cate_scaler_dict[cate_id] = construct_cheat_dict(result_list=list(set(cate_column_dict[cate_id])))
            # cate_column_count_dict[cate_id] = len(list(set(cate_column_dict[cate_id])))
            # 最大数设置embedding, 方便后续调优
            cate_column_count_dict[cate_id] = max(list(map(lambda x: int(x), cate_column_dict[cate_id])))

        training_data = ListDataset([{FieldName.TARGET: x['y'],
                                      FieldName.START: x['start'],
                                      FieldName.ITEM_ID: [x['item_id']],
                                      # FieldName.FEAT_DYNAMIC_REAL: np.array([x['item_id'] for _ in range(len(x['y']))]).reshape((-1, 1)),
                                      FieldName.FEAT_STATIC_CAT: [x[string_idx] for string_idx in cate_column]} for x in
                                     train_sample_list],
                                    freq="10min")

        validating_data = ListDataset([{FieldName.TARGET: x['y'],
                                        FieldName.START: x['start'],
                                        FieldName.ITEM_ID: [x['item_id']],
                                        # FieldName.FEAT_DYNAMIC_REAL: [x['item_id'] for _ in range(len(x['y']))],
                                        FieldName.FEAT_STATIC_CAT: [x[string_idx] for string_idx in cate_column]} for x
                                       in valid_sample_list],
                                      freq="10min")
        """
        # 作弊表用的, 但是废除了, 因为后面api不能调用
        training_data = ListDataset([{FieldName.TARGET: x['y'],
                                      FieldName.START: x['start'],
                                      FieldName.ITEM_ID: [x['item_id']],
                                      # FieldName.FEAT_DYNAMIC_REAL: np.array([x['item_id'] for _ in range(len(x['y']))]).reshape((-1, 1)),
                                      FieldName.FEAT_STATIC_CAT: [
                                          cate_scaler_dict[string_idx][x[string_idx]] for string_idx in
                                          cate_column]} for x in
                                     train_sample_list],
                                    freq="10min")

        validating_data = ListDataset([{FieldName.TARGET: x['y'],
                                        FieldName.START: x['start'],
                                        FieldName.ITEM_ID: [x['item_id']],
                                        # FieldName.FEAT_DYNAMIC_REAL: [x['item_id'] for _ in range(len(x['y']))],
                                        FieldName.FEAT_STATIC_CAT: [
                                            cate_scaler_dict[string_idx][x[string_idx]] for string_idx in
                                            cate_column]} for x
                                       in valid_sample_list],
                                      freq="10min")
        """

        # 大概看一下embedding的维度大小
        print(f"[Info] The cate {cate_column_count_dict}")

        config = {"epochs": 50, "num_batches_per_epoch": 150,
                  "batch_size": 1024, "context_length": 48, "prediction_length": 48,
                  "lags_seq": [6, 12, 18, 24, 32, 38, 42, 48],
                  "num_cells": 128,
                  "cardinality": [cate_column_count_dict[string_idx] + 1 for string_idx in cate_column],
                  # "cardinality": [2000 for string_idx in cate_column],
                  # "device": torch.device("cuda:4" if torch.cuda.is_available() else "cpu"),
                  "device": torch.device("cpu"),
                  "embedding_dimension": [4 for _ in cate_column], "learning_rate": 1.0e-3}

        total_embedding_dimension = sum(config["embedding_dimension"]) + 2


        print(config)
        trainer = Trainer(epochs=config["epochs"], batch_size=config["num_batches_per_epoch"],
                          device=device, learning_rate=config["learning_rate"],
                          num_batches_per_epoch=config["num_batches_per_epoch"])

        estimator = DeepAREstimator(
            # prediction_length=48,
            freq="10min",
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
            input_size=len(config["lags_seq"]) + total_embedding_dimension + 6,
            embedding_dimension=config["embedding_dimension"],
            use_feat_static_cat=True,
            lags_seq=config["lags_seq"],
            num_cells=config["num_cells"],
            cardinality=config["cardinality"],
            distr_output=StudentTOutput(),
            trainer=trainer)

        # estimator = DeepAREstimator(freq="10min", context_length=config["context_length"],
        #                             prediction_length=config["prediction_length"],
        #                             input_size=len(config["lags_seq"]) + total_embedding_dimension + 6,
        #                             lags_seq=config["lags_seq"], num_cells=config["num_cells"],
        #                             trainer=trainer, cardinality=config["cardinality"],
        #                             embedding_dimension=config["embedding_dimension"], use_feat_static_cat=True,)
        # train model
        print('model training start time:{}'.format(datetime.datetime.now()))
        predictor = estimator.train(training_data=training_data, validation_data=validating_data,
                                    num_workers=8, shuffle_buffer_length=1024)
        print('model trained start time:{}'.format(datetime.datetime.now()))

        # save model
        print('model saving start time:{}'.format(datetime.datetime.now()))
        predictor.serialize(Path(output_model_path))
        print('model saved start time:{}'.format(datetime.datetime.now()))

        cmd = 'cd {} && touch model && tar -czf model.tar.gz model'.format(output_model_path)
        ret, _ = subprocess.getstatusoutput(cmd)
        if ret != 0:
            return False
        return True

    def load_model(self, model_path: str, params: Dict, **kwargs) -> bool:
        """从本地加载模型

        Args:
            model_path (str): 本地模型路径
            params (Dict): 模型输入参数。默认为conf/default.json

        Returns:
            bool: True 成功; False 失败
        """
        # load model
        self.model = PyTorchPredictor.deserialize(Path(model_path),device=device)
        return True

    def predicts(self, sample_list: List[Dict], **kwargs) -> List[Dict]:
        """
        批量预测

        Args:
            sample_list (List[Dict]): 输入请求内容列表
            kwargs:
                __dataset_root_path (str): 本地输入路径
                __output_root_path (str):  本地输出路径

        Returns:
            List[Dict]: 输出预测结果列表
        """
        # 请将输出图片请放到output_path下
        # input_path = kwargs.get('__dataset_root_path')
        # output_path = kwargs.get('__output_root_path')
        # sample_list [{'':''},{'':''}]
        # 根据输入内容，填写计算的答案
        # inferencing data
        print('inferencing data:{}'.format(datetime.datetime.now()))

        cate_column = ["item_id", "app_id", "zone_id"][0:2]

        testing_data = ListDataset([{FieldName.TARGET: x['y'],
                                     FieldName.START: x['start'],
                                     FieldName.ITEM_ID: [x['item_id']],
                                     # FieldName.FEAT_DYNAMIC_REAL: [x['item_id'] for _ in range(len(x['y']))],
                                     FieldName.FEAT_STATIC_CAT: [x[string_idx] for string_idx in cate_column]} for x in valid_sample_list],
                                     freq="10min")

        # test_data = ListDataset([{FieldName.TARGET: x['y'],
        #                           FieldName.START: x['start'],
        #                           FieldName.FEAT_STATIC_CAT: [x['item_id'][0]]} for x in sample_list],
        #                         freq='10min')
        predicts = list(self.model.predict(testing_data))
        ret = [{'prediction': [float(i) for i in list(pred.samples.mean(axis=0).reshape(-1, ))],} for pred in predicts]

        print('inferenced data:{}'.format(datetime.datetime.now()))
        return ret
