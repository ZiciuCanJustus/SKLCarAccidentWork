from gluonts.dataset.pandas import PandasDataset
from json_utils import generate_time_list
import pandas as pd
import datetime

def convert_sample_dataframe(sample, pd_time=True):
    """
    :param sample: 处理好了的样本数据
    :return: 一个dataframe
    """

    value_list = sample["y"]
    time_list = generate_time_list(start_time=sample["start"], time_length=len(value_list), min_freq=10)
    if pd_time:
        time_list_str = list(map(lambda x: datetime.datetime.strftime(x, "%Y-%m-%d %H:%M:%S"), time_list))
        time_list = list(map(lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S"), time_list_str))
    single_dataframe = pd.DataFrame({"time": time_list, "y": value_list})
    single_dataframe["item_id"] = sample["item_id"]
    single_dataframe["app_id"] = sample["app_id"]
    single_dataframe["zone_id"] = sample["zone_id"]
    return single_dataframe

# def convert_sample_list(sample):




if __name__ == "__main__":

    """
    测试json文件的读取, 以及我们读取的情况
    """

    import os
    import json
    from json_utils import preprocess_outlier_data

    data_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "\data\\train.jsonl"
    print(data_path)
    iter_times = 0

    train_samples = []
    dataframe_list = []

    with open(data_path) as f:
        for line in f:
            sample = json.loads(line)
            iter_times = iter_times + 1
            # train_samples.append(sample)
            # print(sample)
            new_sample = preprocess_outlier_data(sample=sample, residual_substitude=True, last_week_number=2)
            train_samples.append(new_sample)
            temp_dataframe = convert_sample_dataframe(new_sample)
            dataframe_list.append(temp_dataframe)
            print(f"[Info] Iter times {iter_times}")
            if iter_times == 2:
                break
    total_dataframe = pd.concat(dataframe_list, axis=0).sort_values(by=["item_id", "time"], ascending=[True, True])

    mxts_dataset = PandasDataset.from_long_dataframe(total_dataframe, target="y", item_id="item_id",
                                                     timestamp="time", freq="10min",
                                                     feat_static_cat=["app_id", "zone_id"])

    from gluonts.dataset.split import split

    print("处理完毕")
    print(total_dataframe)
    print(mxts_dataset)

    training_dataset, testing_dataset = split(mxts_dataset, date=pd.Period("2022-04-15 00:00:00", freq="10min"))

    print("干活完成!")

    # _, test_template = split(mxts_dataset, date=pd.Period("2015-04-07 00:00:00", freq="1H"))

    # print(train_samples[0])
