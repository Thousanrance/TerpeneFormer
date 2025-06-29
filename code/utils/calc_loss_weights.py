import pandas as pd

train_csvs = ['/path/to/Terpene-former/data/Retro-tero/random_split/train_enz_rxnfp.csv',
              '/path/to/Terpene-former/data/Retro-tero/tanimoto_split_thres0.6/train_enz_rxnfp.csv',
              '/path/to/Terpene-former/data/Retro-tero/tanimoto_split_thres0.4/train_enz_rxnfp.csv']

num_classes = 10

for train_csv in train_csvs:
    print(f"Processing {train_csv}...")

    # 读取 CSV 文件
    df = pd.read_csv(train_csv)

    # 统计每个类别的数量
    class_counts = df[f'rxnfp_{num_classes}_euc'].value_counts().sort_index()
    print("每个类别的数量：")
    print(class_counts)

    # 计算权重（常用公式：总样本数 / (类别数 * 每个类别的样本数)）
    total = len(df)
    num_classes = class_counts.shape[0]
    weights = total / (num_classes * class_counts)
    print("每个类别的权重：")
    print(weights)

    # 如果需要转为 list 或 tensor，可如下
    weight_list = weights.tolist()
    print(" ".join(str(w) for w in weight_list))


