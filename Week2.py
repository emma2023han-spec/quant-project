import csv


def mean(nums):
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


def std(nums):
    if len(nums) < 2:
        return 0.0
    average = mean(nums)
    total = 0.0
    for item in nums:
        total += (item - average) ** 2
    return (total / (len(nums) - 1)) ** 0.5


def correlation(list1, list2):
    if not list1 or not list2 or len(list1) != len(list2):
        return 0.0
    n = len(list1)
    avg1, avg2 = mean(list1), mean(list2)
    num = sum((list1[i] - avg1) * (list2[i] - avg2) for i in range(n))
    den = (sum((x - avg1) ** 2 for x in list1) * sum((y - avg2) ** 2 for y in list2)) ** 0.5
    return num / den if den != 0 else 0.0


def calculate_features():
    dates, jpm, vix, rates = [], [], [], []

    # 1. 读取 JPM 和 VIX 数据
    with open("jpm_vix_raw.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(row['Date'])
            jpm.append(float(row['JPM']))
            vix.append(float(row['VIX']))

            # 2. 读取国债收益率数据 (DGS10)
    rate_dict = {}
    with open("dgs10_raw.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                date_val = row[0].strip()
                rate_val = row[1].strip()
                # 过滤掉空字符串、空格和 <null>
                if date_val and rate_val and date_val != '<null>' and rate_val != '<null>':
                    try:
                        # 用 try-except 兜底，防止任何意外格式导致崩溃
                        rate_dict[date_val] = float(rate_val)
                    except ValueError:
                        pass # 遇到无法转换的行就跳过

    # 3. 按日期对齐国债数据
    for d in dates:
        rates.append(rate_dict.get(d, 0.0))

    # 4. 初始化特征字典
    features = {
        "Date": dates, "JPM_Return": [0.0] * len(dates),
        "Vol_21d": [0.0] * len(dates), "VIX_Change": [0.0] * len(dates),
        "Corr_21d": [0.0] * len(dates), "Rate_Mom_5d": [0.0] * len(dates),
        "Sentiment": [0.5] * len(dates)
    }

    # 5. 循环计算特征
    for i in range(len(dates)):
        if i > 0:
            features["JPM_Return"][i] = (jpm[i] - jpm[i - 1]) / jpm[i - 1]
            features["VIX_Change"][i] = (vix[i] - vix[i - 1]) / vix[i - 1]
        if i >= 20:
            jpm_win = features["JPM_Return"][i - 20:i + 1]
            features["Vol_21d"][i] = std(jpm_win) * (252 ** 0.5)
            vix_win = vix[i - 20:i + 1]
            features["Corr_21d"][i] = correlation(jpm_win, vix_win)
            min_v, max_v = min(vix_win), max(vix_win)
            if max_v != min_v:
                features["Sentiment"][i] = 1 - (vix[i] - min_v) / (max_v - min_v)
        if i >= 5:
            features["Rate_Mom_5d"][i] = rates[i] - rates[i - 5]

    # 6. 输出结果到 feature_output.csv
    with open("feature_output.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(features.keys())
        for i in range(len(dates)):
            writer.writerow([features[key][i] for key in features])
    print("特征计算完成，生成了 feature_output.csv")


if __name__ == "__main__":
    calculate_features()  # 注意这里不需要传文件名了，代码里写死了
