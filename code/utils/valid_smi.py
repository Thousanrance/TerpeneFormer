import pandas as pd
from rdkit import Chem
import os

def is_valid_smiles(smiles):
    """检查SMILES字符串是否有效"""
    return Chem.MolFromSmiles(smiles) is not None

def calculate_validity(df, columns, topk_values):
    """计算每行的前k个SMILES字符串的有效性"""
    for topk in topk_values:
        df[f'valid_top{topk}'] = df.apply(lambda row: sum(is_valid_smiles(row[col]) for col in columns[:topk]), axis=1)
    return df

def calculate_overall_validity(df, topk_values):
    """计算每个topk的总体有效率"""
    total_rows = len(df)
    overall_validity = {}
    for topk in topk_values:
        valid_count = df[f'valid_top{topk}'].sum()
        overall_validity[f'overall_validity_top{topk}'] = valid_count / (total_rows * topk)
    return overall_validity

def valid_result():
    # 读取CSV文件
    input_csv = '/amax/data/lishuaixin/ckpts/retroformer_multi_modal/retroformer/ckpts_finetune_enz_rxnfp/euc/thres_0.4_frac_0.7_10_untyped/predictions_model_265000.ptTOP10_update_invalid_token.csv'
    df = pd.read_csv(input_csv)

    # 定义预测列和topk值
    pred_columns = [f'pred_{i}' for i in range(1, 11)]
    topk_values = [1, 3, 5, 10]

    # 计算有效性并更新数据框
    df = calculate_validity(df, pred_columns, topk_values)

    # 计算每个topk的总体有效率
    overall_validity = calculate_overall_validity(df, topk_values)

    # 打印总体有效率
    for topk, validity in overall_validity.items():
        print(f'{topk}: {validity:.2%}')

    # 保存更新后的CSV文件
    folder = os.path.dirname(input_csv)
    output_csv = folder + '/valid_smi.csv'
    df.to_csv(output_csv, index=False)


if __name__=='__main__':
    valid_result()
    # print(Chem.MolFromSmiles('CN1C[C@H](C(=O)O)=C[C@@H]2c3cccc4[nH]cc(c34)C[C@H]21'))
    # a = Chem.MolFromSmiles('CN1C[C@H](C(=O)O)=C[C@@H]2c3cccc4[nH]cc(c34)C[C@H]21') is not None
    # print(a)