import os
import json
import numpy as np
import pandas as pd


def process_nan(x):
    if np.isnan(x):
        return 0.0
    else:
        return float(x)


def boxplot_wash(data):
    """
    :param data: 一条数据 list[]
    :return: 清洗好的boxplot
    """

    def _substitude_logic(data, up_value, down_value):
        if (data < up_value) & (data > down_value):
            return data
        elif data > up_value:
            return up_value
        elif data < down_value:
            return down_value

    # 分位数结果
    quantile_75, quantile_25 = np.quantile(data, q=0.75), np.quantile(data, q=0.25)
    iqr = quantile_75 - quantile_25

    # 上下四分位结果
    upper_bound, lower_bound = quantile_75 + 1.5 * iqr, quantile_25 - 1.5 * iqr

    # 替换逻辑过程
    new_data = list(map(lambda x: _substitude_logic(data=x, up_value=upper_bound, down_value=lower_bound), data))
    return new_data


def extract_one_day_wave(past_dataframe, last_week):
    """
    :param past_dataframe: 过去的dataframe 全量数据也可以
    :param last_week: 要拿走完整的几周数据
    :return: 一个周期波形的dict {0: 0.73737677, 10: 0.7808410607142857, 20: 0.677642615, 30: 0.589311462142857, ...., }每天绝对时间对应地波形大小
    """
    past_dataframe = past_dataframe.sort_values(by=["time"], ascending=[True])

    # 完整两周数据
    upper_date = past_dataframe.at[past_dataframe.shape[0] - 1, "time"] - pd.to_timedelta(1, unit="day")
    lower_date = upper_date - pd.to_timedelta(int(last_week * 7), unit="days") + pd.to_timedelta(10, unit="minute")

    # 完整两周dataframe
    past_dataframe = past_dataframe[
        (past_dataframe['time'] >= lower_date) & (past_dataframe['time'] <= upper_date)].sort_values(by=["time"],
                                                                                                     ascending=[
                                                                                                         True]).reset_index(
        drop=True)

    # 提取完整两周波形作弊表
    new_waveline = past_dataframe["label"].values.reshape((int(last_week * 7), 144))
    wave_list = np.mean(new_waveline, axis=0)

    # 结果的json
    time_list = list(map(lambda x: x * 10, range(144)))
    result_json = json.dumps(dict(zip(time_list, wave_list)), ensure_ascii=False)

    return result_json


def extract_wave_line(sample, last_week_number):
    label_list = sample["y"]
    start_time_pd = pd.to_datetime(sample["start"], format="%Y-%m-%d %H:%M:%S")
    time_index = list(map(lambda x: start_time_pd + pd.to_timedelta(10 * x, unit="minute"), range(len(label_list))))
    new_dataframe = pd.DataFrame({"time": time_index, "label": label_list})
    temp_dict = extract_one_day_wave(past_dataframe=new_dataframe, last_week=last_week_number)
    return temp_dict


def item_residual(value, minute_number, cheat_dict):
    return value - cheat_dict[str(minute_number)]


def generate_residual(sample, wave_json):
    label_list = sample["y"]
    start_time_pd = pd.to_datetime(sample["start"], format="%Y-%m-%d %H:%M:%S")
    time_index = list(map(lambda x: start_time_pd + pd.to_timedelta(10 * x, unit="minute"), range(len(label_list))))
    new_dataframe = pd.DataFrame({"time": time_index, "label": label_list})
    new_dataframe["minute_value"] = new_dataframe['time'].apply(lambda x: x.minute)

    wave_dict = json.loads(wave_json)
    new_dataframe["residual_value"] = new_dataframe.apply(lambda x: item_residual(value=x["label"],
                                                                                  minute_number=x["minute_value"],
                                                                                  cheat_dict=wave_dict),
                                                          axis=1)

    return new_dataframe["residual_value"].values.reshape((-1)).tolist()


def preprocess_outlier_data(sample, residual_substitude, last_week_number=2):
    value_list = sample["y"]
    # 标签信息处理nan
    value_list = list(map(lambda x: process_nan(x), value_list))

    # 标签信息做boxplot
    value_list = boxplot_wash(data=value_list)
    sample['y'] = value_list

    if residual_substitude:
        # 预处理时间的东西
        wave_dict_value = extract_wave_line(sample=sample, last_week_number=last_week_number)
        sample["y"] = generate_residual(sample=sample, wave_json=wave_dict_value)

    return sample


def sample_train_validate(sample, validation_length, box_plot):
    """
    :param sample: 一个读取的sample样本
    :param predict_length: 预测的长度
    :return:
    """

    if box_plot:
        sample = preprocess_outlier_data(sample=sample, residual_substitude=False, last_week_number=2)

    start_time = sample["start"]
    value_list = sample["y"]



    # 名字命名
    app_id, zone_id, item_id = sample["app_id"], sample["zone_id"], sample["item_id"]

    # 训练集&测试集时间长短
    train_value_list = value_list[:-validation_length]
    valid_value_list = value_list

    # 字典dict
    train_dict, valid_dict = dict(), dict()

    # 共有函数
    # train_dict["app_id"], train_dict["zone_id"], train_dict["item_id"], train_dict["start"] = f'app_{app_id}', f'zone_{zone_id}', f'item_{item_id}', start_time
    # valid_dict["app_id"], valid_dict["zone_id"], valid_dict["item_id"], valid_dict["start"] = f'app_{app_id}', f'zone_{zone_id}', f'item_{item_id}', start_time

    # 不搞embedding那一套, 直接干活
    train_dict["app_id"], train_dict["zone_id"], train_dict["item_id"], train_dict["start"] = app_id, zone_id, item_id, start_time
    valid_dict["app_id"], valid_dict["zone_id"], valid_dict["item_id"], valid_dict["start"] = app_id, zone_id, item_id, start_time

    # 各自函数
    train_dict["y"], valid_dict["y"] = train_value_list, valid_value_list
    return train_dict, valid_dict



def construct_cheat_dict(result_list):
    """
    :param result_list:转换作弊表
    :return:一个作弊字典
    """
    # 升序排序
    new_list = sorted(result_list, reverse=False)
    # 作弊表构建
    embedding_idx = list(range(len(result_list)))
    cheat_sheet = dict(zip(new_list, embedding_idx))
    return cheat_sheet


if __name__ == "__main__":
    print("[Info] we are doing data!")
