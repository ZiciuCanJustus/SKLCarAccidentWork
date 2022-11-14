from typing import Dict, List
import json
from gluonts.dataset.common import ListDataset
import subprocess
from gluonts.dataset.field_names import FieldName
from typing import Callable, Iterable, Iterator, List
import torch
from pts.model.deepar import DeepAREstimator
from pts.model.n_beats import NBEATSEstimator
from pts import Trainer
import os
from pathlib import Path
from gluonts.torch.model.predictor import PyTorchPredictor
# import torch.distributed as dist
import numpy as np
import datetime
from utils.json_utils import sample_train_validate, construct_cheat_dict
from sklearn.preprocessing import LabelEncoder
from pts.modules import StudentTOutput, StudentTMixtureOutput, NormalOutput, FlowOutput

# device = torch.device("cpu")
import joblib
from torch.utils.data import DataLoader, Dataset
from utils.torch_dataset_utils import construct_dataset
from modelFormer.Informer import ModifiedInformerModel
from utils.model_utils import train_torch_model, inference_model
import random


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model_name = "informer_model.pth"
scaler_name = "informer_scaler.pkl"
config_name = "informer_configs.pkl"
# CATE_COL = cate_column = ["item_id", "app_id", "zone_id"][1:2]
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False





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

        seed_torch(seed=42)
        train_data_path = os.path.join(input_data_path, 'train.jsonl')

        print(device)
        train_samples = []

        # 训练样本、验证样本的list
        train_sample_list, valid_sample_list = list(), list()
        # 全部样本集合
        sample_list = list()


        # cate类列的名字[选择哪些]
        cate_column = ["item_id", "app_id", "zone_id"]# [1:2]


        iter_time = 0

        item_list, app_list, zone_list = [], [], []
        with open(train_data_path) as f:
            for line in f:
                # 读取sample结果
                sample = json.loads(line)
                sample_list.append(sample)
                sample["y"] = sample["y"][-144 * 14:]

                item_list.append(sample['item_id'])
                app_list.append(sample['app_id'])
                zone_list.append(sample['zone_id'])

                train_sample, valid_sample = sample_train_validate(sample=sample, validation_length=144 * 2, box_plot=True)
                train_sample_list.append(train_sample)
                valid_sample_list.append(valid_sample)

                sample = json.loads(line)
                train_samples.append(sample)

                iter_time = iter_time + 1
                if iter_time >= 10:
                    break

        batch_size = 256


        # 构建数据集、以及scaler的东西

        train_dataset, scaler_dict, unique_element = construct_dataset(sample_list=train_sample_list,
                                                                       cate_list=cate_column,
                                                                       item_list=item_list, app_list=app_list,
                                                                       zone_list=zone_list, device=device)
        print("[Info] We finished training!")

        # 测试样本的东西
        valid_dataset, _, _ = construct_dataset(sample_list=valid_sample_list,
                                                cate_list=cate_column,
                                                item_list=item_list, app_list=app_list,
                                                zone_list=zone_list, device=device)
        print("[Info] We finished validating!")

        configs = {"encoder_length": 144, "pred_length": 48, "dropout": 0.3, "model_dimension": 128,
                   "data_dimension": 1, "epochs": 3,
                   "cardinality": unique_element, "scale_factor": 10, "head_num": 8, "ff_dimension": 16,
                   "activated_function": "gelu", "encoder_layers": 6, "decoder_layers": 6, "device": device,
                   "batch_size": batch_size, "learning_rate": 1.0e-3,
                   "output_dimension": 1, "distillation": True, "revIn": True}

        # 设置loader以及batch size
        train_loader, valid_loader = DataLoader(dataset=train_dataset, batch_size=configs["batch_size"], shuffle=True), DataLoader(dataset=valid_dataset, batch_size=configs["batch_size"], shuffle=True)
        informer_model = ModifiedInformerModel(configs=configs).to(device=configs["device"])

        train_torch_model(train_loader=train_loader, model=informer_model,
                          valid_loader=valid_loader, epoch_number=configs["epochs"],
                          model_path=output_model_path, learning_rate=configs["learning_rate"],
                          predict_length=configs["pred_length"], model_name=model_name)


        # 存储scaler
        joblib.dump(scaler_dict, output_model_path + '/' + scaler_name)
        joblib.dump(configs, output_model_path + '/' + config_name)


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
        # 装载模型
        self.scaler_dict = joblib.load(model_path + scaler_name)
        self.configs_dict = joblib.load(model_path + config_name)
        # load model
        self.model = ModifiedInformerModel(configs=self.configs_dict).to(device=self.configs_dict["device"])
        hisotry_params = torch.load(model_path + '/' + model_name)
        self.model.load_state_dict(hisotry_params)
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
        print('[Info] Inferring data time:{}'.format(datetime.datetime.now()))
        start_token_list, item_list, app_list, zone_list = [], [], [], []
        batch_size = 128

        for sample in sample_list:
            # 读取sample结果

            # sample_list.append(sample)
            start_token_list.append(np.array(sample["y"], dtype=np.float32).reshape((1, -1, 1)))
            item_list.append(np.array(self.scaler_dict['item_id'].transform([sample['item_id']])[0], dtype=np.int64).reshape((-1, 1)))
            app_list.append(np.array(self.scaler_dict['app_id'].transform([sample['app_id']])[0], dtype=np.int64).reshape((-1, 1)))
            zone_list.append(np.array(self.scaler_dict['zone_id'].transform([sample['zone_id']])[0], dtype=np.int64).reshape((-1, 1)))

        start_token_array = np.concatenate(start_token_list, axis=0)
        item_array = np.concatenate(item_list, axis=0)
        app_array = np.concatenate(app_list, axis=0)
        zone_array = np.concatenate(zone_list, axis=0)

        start_token_tensor = torch.tensor(start_token_array, dtype=torch.float32, device=device)
        item_tensor = torch.tensor(item_array, dtype=torch.int64, device=device)
        app_tensor = torch.tensor(app_array, dtype=torch.int64, device=device)
        zone_tensor = torch.tensor(zone_array, dtype=torch.int64, device=device)

        result_array = inference_model(model=self.model,
                                        start_token_tensor=start_token_tensor,
                                        item_tensor=item_tensor,
                                        app_tensor=app_tensor,
                                        zone_tensor=zone_tensor,
                                        batch_size=batch_size,
                                        predict_length=48).squeeze(-1).cpu().numpy()
        ret = [{'prediction': result_array[x_axis, :].reshape((-1)).tolist(),} for x_axis in range(result_array.shape[0])]

        return ret

