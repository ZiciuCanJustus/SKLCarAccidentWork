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
    @numba.jit()
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
