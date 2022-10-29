import numpy as np
import pandas as pd
from statsmodels.nonparametric import smoothers_lowess
import datetime


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
    


if __name__ == "__main__":
    print("[Info] 进行boxplot替换异常值操作!")














