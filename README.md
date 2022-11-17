# ATEC绿色计算赛道1比赛经验总结(国重碰碰车翻车案例)
## 写在前面
国重碰碰车车队18/40(成绩不是太好，但是感觉还行)的整体经验感觉:
* 首先要注意比赛**官方文档**, 这次比赛不是直接提交固定格式的csv, 而是直接交docker, 他去运行docker。所以这个docker啥的基础操作得注意一下  
* 其次就是, 他那个官方给的代码, 要理清思路, **主要是理清思路**, 他说不能用历史均值特征什么的, 但是他后台调用的程序来看, 你完全可以存一个**json文件**到固定地址, 然后直接干活!
* 再次是良好的代码处理链路, 不然队友之间合作很不好合作。代码一定要凸显抽象类和层次性, **写好本地测试文件，如果他官方要求交docker**
* 最后就是一定要留够充分的时间，比赛前期我被抓去**解决重大战略需求**，比赛后期队友被抓去**解决重大战略需求**，我们真正开始搞这个比赛的时间不是特别多，所以时间比较紧凑，最后几天干通宵去摸代码、交文件去摸奖

## 前处理链路
* 深度学习模型, **官方给出的demon可能不是特别简洁、明了、乃至代码是一坨shit**，但是可以从这里理解比赛代码的执行逻辑。
这个时候建议把自己的链路, 从dataset到dataloader按照自己的习惯写一遍，保留抽象类的特点，不然**后面可能坑队友**
官方代码文件:
接下来进行示例说明：
```python
from typing import Dict, List
import json
from gluonts.dataset.common import ListDataset
import subprocess
from gluonts.dataset.field_names import FieldName
from typing import Callable, Iterable, Iterator, List
import torch
from pts.model.deepar import DeepAREstimator
from pts.model.lstnet import LSTNetEstimator
from pts.model.n_beats import NBEATSEstimator

from pts import Trainer
import os
from pathlib import Path
from gluonts.torch.model.predictor import PyTorchPredictor
# import torch.distributed as dist
import numpy as np
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlgSolution():
    def __init__(self):
        pass

    def train_model(self, input_data_path: str, output_model_path: str, params: Dict, **kwargs) -> bool:
        # """使用数据集训练模型

        # Args:
        #     input_data_path (str): 本地输入数据集路径
        #     output_model_path (str): 本地输出模型路径
        #     params (Dict): 训练输入参数。默认为conf/default.json

        # Returns:
        #     bool: True 成功; False 失败
        # """
        # load pretrained model if any
        # self.model = load_from_pretrained()
        # reading and processing data
        train_data_path = os.path.join(input_data_path, 'train.jsonl')
        print(device)
        train_samples = []
        with open(train_data_path) as f:
            for line in f:
                sample = json.loads(line)
                train_samples.append(sample)

        training_data = ListDataset(
            [{FieldName.TARGET: x['y'],
              FieldName.START: x['start'],
              FieldName.FEAT_STATIC_CAT: [x['item_id'][0]],
              # FieldName.FEAT_DYNAMIC_CAT:np.array([]).T
              } for x in train_samples],
            freq='10min',
        )
        # parameter setting and estimator setting
        config = {
            "epochs": 50,
            "num_batches_per_epoch": 150,
            "batch_size": 2048,
            "context_length": 48,
            "prediction_length": 48,
            "lags_seq": [6, 12, 18, 24, 32, 38, 42, 48, 96],
            # "lags_seq":[32, 38, 42, 48, 96,144],
            "num_cells": 128,
            "cardinality": [2000],
            "embedding_dimension": [5],
            "learning_rate": 1e-3,
        }
        print(config)
        trainer = Trainer(
            epochs=config["epochs"],
            batch_size=config["num_batches_per_epoch"],
            device=device,
            learning_rate=config["learning_rate"],
            num_batches_per_epoch=config["num_batches_per_epoch"])

       
        estimator = DeepAREstimator(freq="10min",
                                    context_length=config["context_length"],
                                    prediction_length=config["prediction_length"],
                                    input_size=22,
                                    lags_seq=config["lags_seq"],
                                    num_cells=config["num_cells"],
                                    trainer=trainer,
                                    cardinality=config["cardinality"],
                                    embedding_dimension=config["embedding_dimension"],
                                    use_feat_static_cat=True,
                                    )
        
        # train model
        print('model training start time:{}'.format(datetime.datetime.now()))
        predictor = estimator.train(training_data=training_data, num_workers=8, shuffle_buffer_length=1024)
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
        self.model = PyTorchPredictor.deserialize(Path(model_path), device=device)
        return True

    def predicts(self, sample_list: List[Dict], **kwargs) -> List[Dict]:
        """批量预测

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
        test_data = ListDataset(
            [{FieldName.TARGET: x['y'],
              FieldName.START: x['start'],
              FieldName.FEAT_STATIC_CAT: [x['item_id'][0]],
              # FieldName.FEAT_DYNAMIC_CAT:np.array([]).T
              } for x in sample_list],
            freq='10min',
        )
        predicts = list(self.model.predict(test_data))
        ret = [{
            'prediction': [float(i) for i in list(pred.samples.mean(axis=0).reshape(-1, ))],
        } for pred in predicts]
        print('inferenced data:{}'.format(datetime.datetime.now()))

        return ret
```
    * 首先这里他的train函数是为了读取json路径的文件，然后用sample去扩展。扩展好了之后送去给一个gluonts的数据集类进行操作
    * 操作好了之后送进去训练模型、然后存储模型参数
    * 然后走加载，把self.model加载进去，最后test预测

基本的代码思路走通了，然后我们注意到以下几个问题：  
```
    * 他会调用train, train里面有预处理函数处理json输入，这个时候我们可以在这里动手脚，写好处理链路。包括一个一个json样本的处理链路、然后后面装载数据集的处理链路
    * trian完可以存储模型，所以我们这里理论上是可以把类似pkl的存储文件，存储必要信息再load那一步装载进去model的
    * 最后就是服务这里，也要处理json文件，这里可能有坑，我们在干活的时候发现一个问题——我们的操作可能会面临nan数值问题，最后打分有问题，这里要注意写后处理链路解决
```
然后这里不放心的几点，gluonts那个ListDataset抽样是在时间轴上面滚几圈，然后干活。
而且按照google TFT对time series的特征的说明，我们有cate特征(nn.Embedding)和continuous特征(nn.Linear)，
还有动态静态区分，他没有用到ListDataset的几个接口，说明比赛官方凑了个东西就上去完事儿了：
```
FEAT_STATIC_CAT = "feat_static_cat"
FEAT_STATIC_REAL = "feat_static_real"
FEAT_DYNAMIC_CAT = "feat_dynamic_cat"
FEAT_DYNAMIC_REAL = "feat_dynamic_real"
PAST_FEAT_DYNAMIC_REAL = "past_feat_dynamic_real"
FEAT_DYNAMIC_REAL_LEGACY = "dynamic_feat"
```
而且，ListDataset抽样的方式是在时间轴上面随机滚去抽，不是特别安全。
所以最后**建议自己写dataset**这是我的dataset：
```python
import numpy as np
import json
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from itertools import chain
import numba
import warnings
# 过滤掉numba的影响
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
# def down_sampling_data

def construct_sample(sample, scaler_dict, cate_col, encoder_steps, decoder_steps):

    # 标签序列
    label_list = sample["y"]
    start_time = sample["start"]

    data_length = len(label_list)

    # 开始的index list
    # print("[Info] We are in the construct_sample function!")
    start_list = list(map(lambda x: x, range(data_length - (encoder_steps + decoder_steps) + 1)))

    # 结束的index list
    encoder_end_list = list(map(lambda x: x + encoder_steps, start_list))
    decoder_end_list = list(map(lambda x: x + encoder_steps + decoder_steps, start_list))

    # 起始序列长度
    start_token_list = list(map(lambda x: np.array(label_list[start_list[x]: encoder_end_list[x]], dtype=np.float32).reshape((1, -1, 1)), range(len(start_list))))
    predict_token_list = list(map(lambda x: np.array(label_list[encoder_end_list[x]: decoder_end_list[x]], dtype=np.float32).reshape((1, -1, 1)), range(len(start_list))))

    # 数据序列
    start_token_list = list(chain(*start_token_list))
    predict_token_list = list(chain(*predict_token_list))

    # print(scaler_dict[cate_col[0]].transform(sample[cate_col[0]]))
    # cate列
    scaled_data = list(map(lambda cate_id: np.array([scaler_dict[cate_id].transform([sample[cate_id]])[0] for _ in range(len(start_list))], dtype=np.int64).reshape((-1, 1)), cate_col))
    result_dict = dict(zip(cate_col, scaled_data))

    result_dict.update({"start_token": start_token_list, "predict_token": predict_token_list})

    return result_dict



class MyDataset(Dataset):
    @numba.jit() #for加速
    def __init__(self, data_list, encoder_steps, decoder_steps, scaler_dict, cate_col, device):
        """
        :param data_list: 全量的datalist文件
        :param encoder_steps: 编码器步长(丢进去encoder的步长)
        :param decoder_steps: 解码器步长(丢进去decoder的步长)
        """

        input_list = []
        output_list = []
        item_list = []
        app_list = []
        zone_list = []


        for sample in data_list:
            result_dict = construct_sample(sample=sample, scaler_dict=scaler_dict,
                                           cate_col=cate_col, encoder_steps=encoder_steps,
                                           decoder_steps=decoder_steps)

            input_list.append(result_dict["start_token"])
            output_list.append(result_dict["predict_token"])

            item_list.append(result_dict[cate_col[0]])
            app_list.append(result_dict[cate_col[1]])
            zone_list.append(result_dict[cate_col[2]])

        self.inputs = torch.from_numpy(np.concatenate(input_list, axis=0)).to(device)
        self.outputs = torch.from_numpy(np.concatenate(output_list, axis=0)).to(device)
        self.items = torch.from_numpy(np.concatenate(item_list, axis=0)).to(device)
        self.apps = torch.from_numpy(np.concatenate(app_list, axis=0)).to(device)
        self.zones = torch.from_numpy(np.concatenate(zone_list, axis=0)).to(device)

    def __getitem__(self, index):
        return self.inputs[index, :, :], self.outputs[index, :, :], self.items[index, :], self.apps[index, :], self.zones[index, :]

    def __len__(self):
        return self.inputs.shape[0]



def construct_dataset(sample_list, cate_list, item_list, app_list, zone_list, device):
    """
    :param sample_list: 样本的list
    :param cate_list: 独立元素的list
    :param item_list: 条目的list
    :param app_list: app的list
    :param zone_list: zone的list
    :param device: 设备
    :return: newDataset, scaler_dict, unique_element
    """
    # 独立元素个数
    item_list, app_list, zone_list = list(set(item_list)), list(set(app_list)), list(set(zone_list))
    unique_element = [len(item_list), len(app_list), len(zone_list)]
    # scaler处理器
    item_scaler, app_scaler, zone_scaler = LabelEncoder().fit(item_list), LabelEncoder().fit(app_list), LabelEncoder().fit(zone_list)

    scaler_list = [item_scaler, app_scaler, zone_scaler]
    scaler_dict = dict(zip(cate_list, scaler_list))

    newDataset = MyDataset(data_list=sample_list, encoder_steps=144, decoder_steps=48, scaler_dict=scaler_dict,
                           cate_col=cate_list, device=device)

    return newDataset, scaler_dict, unique_element

```
这里我强烈建议大家去copy这个[TFT的代码仓库](https://github.com/KalleBylin/temporal-fusion-transformers)
里面那个dataset的写法，能简单很多。
数据做好了剩下就成功了很多，剩下就是一些model，**自己dataset写什么，model forward就要用哪些输入**。这个一定要记住
时间紧急，我们就魔改了一个Informer。细节不展示了

## 线上效率分
这比赛坑爹在一点就是，你这玩意儿有线上效率分，这特喵绷不住。
* 1587个item，然后俩月，past length是144，吐出来未来48个点。你的sample可能几w条，直接扛不住的。
* former模型自己训练可能比较难顶，所以我们线上只滚了3个epoch就把model送去serving了

# 别的队伍刷高分的操作
等我搜到他们的代码再去继续学习一下，我没拿到，有点尴尬
* 简单模型，直接nn拟合数据+embedding就完事儿了
* dataset重写了更简洁
