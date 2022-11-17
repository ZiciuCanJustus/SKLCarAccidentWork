import numpy as np
import pandas as pd
# /data/anonym1/Justus/timeSeriesCompetition/atecGreen/demon/atec_cloud_traffic_demo/atec_ts_project
from newAlgoSolution import AlgSolution
import os
import json

import datetime
import sys

"""
主要是为了线上测试algo类对不对写的函数，要把链路走通
"""

def construct_test_sample(sample, last_start, remain_length, predict_length):

    label_list = sample["y"]
    start_time_datetime = datetime.datetime.strptime(sample["start"], "%Y-%m-%d %H:%M:%S")
    time_index = list(map(lambda x: start_time_datetime + x * datetime.timedelta(minutes=10), range(len(label_list))))

    truncation_place = last_start + remain_length + predict_length

    test_time_list, test_label_list = time_index[-truncation_place:], label_list[-truncation_place:]

    # 请求时间
    request_time = datetime.datetime.strftime(test_time_list[0], "%Y-%m-%d %H:%M:%S")
    request_label = test_label_list[:remain_length]

    # 请求list
    request_dict = dict()
    request_dict["y"] = request_label
    request_dict["start"] = request_time

    request_dict["app_id"], request_dict["zone_id"], request_dict["item_id"] = sample["app_id"], sample["zone_id"], sample["item_id"]

    real_label = test_label_list[-predict_length:]

    # print(type(request_dict["app_id"]), type(request_dict["zone_id"]), type(request_dict["item_id"]))

    return request_dict, real_label

if __name__ == "__main__":
    # print(f"[Info] The local file path is {os.getcwd()}")
    input_data_path = "/data/anonym1/Justus/timeSeriesCompetition/atecGreen/"
    output_model_path = "/data/anonym1/Justus/timeSeriesCompetition/atecGreen/model_save"
    params = None
    algo_solution = AlgSolution()
    algo_solution.train_model(input_data_path=input_data_path,
                                            output_model_path=output_model_path,
                                            params=params)


    # test model
    data_path = input_data_path + "/train.jsonl"
    test_sample_list, label_list = list(), list()

    iter_times = 0

    with open(data_path) as f:
        for line in f:
            # 读取sample结果
            sample = json.loads(line)

            iter_times = iter_times + 1
            # print(f"[Info] The iter times {iter_times}")

            request_dict, real_label = construct_test_sample(sample=sample, last_start=10,
                                                             remain_length=144, predict_length=int(8 * 6))



            test_sample_list.append(request_dict)
            label_list.append(real_label)

            sys.exit(0)

            # if iter_times >= 10:
            #     break

    print("[Info] We succeed finished!")

    algo_solution.load_model(model_path=output_model_path, params=None)
    print("[Info] We succeed finished the loading model!")
    results = algo_solution.predicts(sample_list=test_sample_list)
    print(f"[Info] we print the first element for prediction:\n {results[0]}")
    print(f"[Info] we print the first element for real:\n {label_list[0]}")
    new_results = list()
    for x in range(len(results)):
        temp_dict = results[x]
        temp_dict["real"] = label_list[x]
        new_results.append(temp_dict)
    # new_results = [results[x]["real"]= for x in range(len(results))]

    import numpy as np
    print(f"[Info] we print the first element:\n {new_results[0]}")
    result_np = np.array(new_results)
    np.save(output_model_path + '/result.npy', result_np)
    print("[Info] We finished the saving")
    # /data/anonym1/Justus/timeSeriesCompetition/atecGreen/train.jsonl
    # /data/anonym1/Justus/timeSeriesCompetition/atecGreen/model_save
