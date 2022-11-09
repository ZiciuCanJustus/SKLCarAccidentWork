import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import datetime
import statsmodels.api as sm
from itertools import chain
import json

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

def extract_daily_waveline(time_list, data_list):
    """
    :param time_list: 时间周
    :param data_list: 数据
    :return: 一个周期的波形
    """
    try:
        time_list_date_time = list(map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), time_list))
    except:
        time_list_date_time = time_list

    start_date, end_date = time_list_date_time[0] + datetime.timedelta(days=1), time_list_date_time[-1] - datetime.timedelta(days=1)
    delta_days = (end_date - start_date).days
    start_date_str = datetime.datetime.strftime(start_date, "%Y-%m-%d")
    end_date_str = datetime.datetime.strftime(end_date, "%Y-%m-%d")

    total_dataframe = pd.DataFrame(data={"time": time_list, "data": data_list})
    selected_dataframe = total_dataframe[(total_dataframe['time']>=datetime.datetime.strptime(start_date_str, "%Y-%m-%d"))
                                         & (total_dataframe['time']<=datetime.datetime.strptime(end_date_str, "%Y-%m-%d"))]
    selected_value = np.array(selected_dataframe["data"]).reshape((delta_days, -1))
    wave_list = np.mean(selected_value, axis=0).reshape((-1)).tolist()

    return wave_list

def seasonal_lowess(data, period, def_bandwidth=0.3, coefficient=-2.0, max_bandwidth=0.75):
    """
    :param data:
    :param period:
    :param def_bandwidth:
    :param def_multiplier:
    :param max_bandwidth:
    :return:
    """
    data_length = len(data)
    if period > 0:
        bandwidth = max_bandwidth if (period * coefficient / data_length) else (period * coefficient / data_length)
    else:
        bandwidth = def_bandwidth
    x_list = list(range(data_length))
    lowess = sm.nonparametric.lowess
    lowess_value = lowess(data, x_list, frac=bandwidth)[:, -1].reshape((-1)).tolist()
    return lowess_value

def subsequence_function(data, index_location, period_num, debug=False):
    """
    每个周期位置的信息
    :param data: 数据list
    :param index_location: 指标位置
    :param period_num: 周其大小
    :param debug: 是否debug
    :return: 返回每个周期波形位置的信息
    """
    data_length = len(data)
    repeated_time, residual_value = data_length // period_num, data_length % period_num
    idx_list = list(map(lambda x: x * period_num + index_location, range(repeated_time)))
    if (index_location + 1) <= residual_value:
        idx_list.append(repeated_time * period_num + index_location)

    selected_data = list(map(lambda x: data[x], idx_list))
    total_length = len(idx_list)
    average_value = np.sum(selected_data) / total_length
    if debug:
        print(f"[Information] 全部长度是: {total_length}")
        print(f"[Information] 选择的指标值是: {idx_list}")
        print(f"[Information] 选择的数据list是: {selected_data}")
    return {"selected_data": selected_data, "selected_length": total_length, "average_value": average_value}

def ma_seasonality(data, period):
    """
    :param data:
    :param period:
    :return:
    """
    if period <= 0:
        return list(range(len(data)))
    else:
        period_pair = list(map(lambda x: subsequence_function(data=data, index_location=x, period_num=period), range(period)))
        result_list = list(map(lambda x: x["average_value"], period_pair))
        total_mean = np.sum(result_list) / period
        final_result_list = list(map(lambda x: x - total_mean, result_list))
        return final_result_list

def seasonality_copy(data, total_length, period):
    """
    :param data:
    :param total_length:
    :param period:
    :return:
    """
    repeated_time, residual_value = total_length // period, total_length % period
    repeated_period_temp = list(map(lambda x: data, range(repeated_time)))
    repeated_period = list(chain(*repeated_period_temp))
    result_period = repeated_period + data[:residual_value]
    return result_period

def remove_seasonality(data, period, debug=False):
    """
    :param data_list:
    :param period:
    :param debug_flag:
    :return:
    """

    data_length = len(data)
    trend = seasonal_lowess(data=data, period=period)
    detrend = list(map(lambda x, y: x - y, data, trend))
    one_period = ma_seasonality(data=detrend, period=period)
    seasonal = seasonality_copy(data=one_period, total_length=data_length, period=period)
    residual = list(map(lambda x, y: x - y, detrend, seasonal))
    if debug:
        return (residual, seasonal)
    else:
        return residual

def seasonality_wave(data, period, bandwidth=1.0/3.0):
    """
    :param data:
    :param period:
    :param bandwidth:
    :return:
    """
    data_new = boxplot_wash(data=data)
    start_number = len(data) % period
    period_wave = list(map(lambda x: subsequence_function(data=data_new[start_number:],
                                                          index_location=x,
                                                          period_num=period,
                                                          debug=False)["average_value"], range(period)))
    lowess_estimator = sm.nonparametric.lowess
    smoothed_value = lowess_estimator(period_wave, list(range(len(period_wave))), frac=bandwidth)[:, 1].reshape((-1)).tolist()
    if len(smoothed_value) == len(period_wave):
        period_wave = smoothed_value

    repeated_period_temp = list(map(lambda x: period_wave, range(len(data)//period)))
    repeated_period = list(chain(*repeated_period_temp))
    result_list = period_wave[-(len(data) - len(repeated_period)):] + repeated_period
    return result_list

def run_lowess_trend(observation, use_boxplot, period, bandwidth):
    """
    :param observation:
    :param use_boxplot:
    :param period:
    :param bandwidth:
    :return:
    """
    data_length = len(observation)
    temp_result = boxplot_wash(data=observation) if use_boxplot else observation
    lowess_estimator = sm.nonparametric.lowess
    x_list = list(range(data_length))
    if period > 0:
        de_seasonal_result = remove_seasonality(data=temp_result, period=period,)
        lowess_value = lowess_estimator(de_seasonal_result, x_list, frac=bandwidth)[:, 1].reshape((-1)).tolist()
    else:
        lowess_value = lowess_estimator(temp_result, x_list, frac=bandwidth)[:, 1].reshape((-1)).tolist()

    return lowess_value


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


if __name__ == "__main__":
    print("[Info] 进行boxplot替换异常值操作!")














