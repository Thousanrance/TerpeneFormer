from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
import torch
import os, json
import random
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.metrics import silhouette_score
from Bio.Cluster import kcluster
# from kneed import KneeLocator


def get_rxn_list(csv_path, only_pro=False):
    df = pd.read_csv(csv_path)
    std_rxn_update_list = df['std_rxn_update'].str.replace('*', '').tolist()
    if only_pro:
        std_rxn_update_list = [rxn.split('>>')[1] for rxn in std_rxn_update_list]
    rxnid_list = df['rxnid'].tolist()
    return std_rxn_update_list, rxnid_list

def get_emb_pt(save_folder, csv_path, only_pro=False):

    if os.path.exists(f'{save_folder}/all_rxnfps.pt'):
        return f'{save_folder}/all_rxnfps.pt', f'{save_folder}/all_rxnids.pt'
    
    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    
    rxns, rxnid_list = get_rxn_list(csv_path, only_pro)
    fps = []
    for rxn in rxns:
        fp = rxnfp_generator.convert(rxn)
        fps.append(fp)
    torch.save(fps, f'{save_folder}/all_rxnfps.pt')
    torch.save(rxnid_list, f'{save_folder}/all_rxnids.pt')
    return f'{save_folder}/all_rxnfps.pt', f'{save_folder}/all_rxnids.pt'

def check_pt(pt_path):
    fps = torch.load(pt_path)
    
    print(len(fps))
    print(len(fps[0]))

def read_pt_files_and_cluster(emb_path, save_folder, distance='euc', max_clusters=30, save_fig=True):
    embeddings = torch.load(emb_path)
    
    embeddings = np.array(embeddings)
    k_values = range(2, max_clusters+1)  # Start from 2 clusters for silhouette score
    inertias = []
    silhouette_scores = []
    for k in k_values:
        if distance=='euc':
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(embeddings, labels, metric='euclidean'))
        elif distance=='cos':
            # 余弦距离的实现似乎有点问题
            labels, inertia_, nfound = kcluster(embeddings, k, dist='u',npass=10)
            silhouette_avg = silhouette_score(embeddings, labels, metric = 'cosine')
            inertias.append(inertia_)
            silhouette_scores.append(silhouette_avg)

    # Identify the elbow point using the "knee" in the SSE curve
    # kneedle = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
    # elbow_k = kneedle.elbow
    # print(f"Identified elbow (optimal number of clusters): {elbow_k}")

    # Plot SSE
    plt.figure()
    # 增大图片大小和页边距
    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel('Number of Clusters', fontsize=18)
    plt.ylabel('Sum of Squared Errors (SSE)', fontsize=18)
    plt.title('Elbow Method for Optimal Number of Clusters', fontsize=20)
    plt.xticks(k_values, rotation=45, fontsize=16) 
    plt.yticks(fontsize=16) 
    if save_fig and save_folder:
        save_path_sse = f'{save_folder}/max_{max_clusters}_{distance}_sse_elbow.png'
        plt.savefig(save_path_sse)
    plt.show()

    # Plot silhouette score
    plt.figure()
    plt.plot(k_values, silhouette_scores, marker='s', color='b')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal Number of Clusters')
    plt.xticks(k_values, rotation=45, fontsize=10) 
    if save_fig and save_folder:
        save_path_silhouette = f'{save_folder}/max_{max_clusters}_{distance}_silhouette.png'
        plt.savefig(save_path_silhouette)
    plt.show()

def kmeans_clustering_with_num_clusters(emb_folder, save_folder, num_clusters,distance='euc'):
    
    embeddings = torch.load(f'{emb_folder}/all_rxnfps.pt')
    embeddings = np.array(embeddings)

    if distance=='euc':
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(embeddings)
        labels = kmeans.labels_
    elif distance=='cos':
        labels, inertia_, nfound = kcluster(embeddings, num_clusters, dist='u',npass=10)

    tero_rxn_id_list = torch.load(f'{emb_folder}/all_rxnids.pt')
    clustered_rxns = [{'tero_rxn_id': tero_rxn_id_list[i], 'cluster_label': int(labels[i])+1} for i in range(len(labels))]

    folder = f'{save_folder}/rxnfp_kmeans_{distance}/clusters_{num_clusters}'
    os.makedirs(folder, exist_ok=True)
    save_json_path = f'{folder}/clustered_rxns.json'
    with open(save_json_path, 'w') as f:
        json.dump(clustered_rxns, f, indent=4)
    
    df = pd.read_json(save_json_path)
    csv_file_path = save_json_path.replace('.json','.csv')
    df.to_csv(csv_file_path, index=False)


    # Reduce the dimensionality of feature vectors using t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(embeddings)
    
    # Visualize the clusters
    plt.figure(figsize=(8, 6))
    for i in range(num_clusters):
        cluster_points = X_tsne[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')
    plt.title('Visualization of Reactions Clusters using t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))  # Adjust legend position
    plt.tight_layout()  # Adjust layout to prevent overlapping
    save_fig_path = f'{folder}/clustered_rxns_TSNE.png'
    plt.savefig(save_fig_path)  # Save the plot as an image
    plt.show()

    return clustered_rxns, csv_file_path

def get_new_dataset(data_path, k_cluster, rxn_cluster_path, kmeans_label='rxnfp', distance='euc', only_pro=False):
    assert str(k_cluster) in rxn_cluster_path
    assert distance in rxn_cluster_path
    folder = os.path.dirname(data_path)
    filename = data_path.split('/')[-1].split('.')[0]
    save_path = folder+f'/{filename}_enz_{kmeans_label}.csv'
    if f'_{kmeans_label}' in filename:
        save_path = folder+f'/{filename}.csv'
    
    if os.path.exists(save_path):
        df1 = pd.read_csv(save_path)
    else:
        df1 = pd.read_csv(data_path)
    df2 = pd.read_csv(rxn_cluster_path)
    merged_df = pd.merge(df1, df2, left_on='rxnid', right_on='tero_rxn_id', how='left')
    col_name = f'{kmeans_label}_{k_cluster}_{distance}'
    if only_pro:
        col_name += '_pro'
    merged_df[col_name] = merged_df['cluster_label']

    columns_to_drop = ['tero_rxn_id', 'cluster_label']
    merged_df.drop(columns=columns_to_drop, inplace=True)
    
    merged_df.to_csv(save_path, index=False)

'''
get_emb_pt()
check_pt()
read_pt_files_and_cluster(emb_path='/amax/data/lishuaixin/retro_NP/terokit/rxn_cls/all_rxnfps.pt',
                            save_folder='/amax/data/lishuaixin/retro_NP/terokit/rxn_cls')
for i in range(5,15):
    kmeans_clustering_with_num_clusters(emb_folder='/amax/data/lishuaixin/retro_NP/terokit/rxn_cls',
                                    num_clusters=i)
read_pt_files_and_cluster(emb_path='/amax/data/lishuaixin/retro_NP/terokit/rxn_cls/all_rxnfps.pt',
                        save_folder='/amax/data/lishuaixin/retro_NP/terokit/rxn_cls',
                        distance='cos')
for i in range(5,15):
    kmeans_clustering_with_num_clusters(emb_folder='/amax/data/lishuaixin/retro_NP/terokit/rxn_cls',
                                    num_clusters=i, distance='cos')
                        
                        '''

def get_diff_csv(in_path, out_path):
    # 创建一个示例DataFrame
    df = pd.read_csv(in_path)

    # 重命名列
    df = df.rename(columns={'rxnfp_10_euc': 'rxnfp_10_euc_gt'})

    # 复制每一行10次并添加新的列
    new_rows = []
    for _, row in df.iterrows():
        for i in range(1, 12):
            new_row = row.copy()
            if i==11:
                new_row['rxnfp_10_euc'] = new_row['rxnfp_10_euc_gt']
                new_row['rxnfp_10_euc_gt'] = str(new_row['rxnfp_10_euc_gt'])+'c'
                new_rows.append(new_row)
            else:
                new_row['rxnfp_10_euc'] = i
                new_rows.append(new_row)

    # print(new_row[:12])
    new_df = pd.DataFrame(new_rows)
    # 保存到新的CSV文件
    new_df.to_csv(out_path, index=False)

def add_new_col_to_csv(in_path, out_path, col_name, col_value):
    df = pd.read_csv(in_path)
    df[col_name] = col_value
    df.to_csv(out_path, index=False)

'''
# get only pro data cluster
get_emb_pt(save_folder='/amax/data/lishuaixin/retro_NP/terokit/rxn_cls_pro', only_pro= True)
read_pt_files_and_cluster(emb_path='/amax/data/lishuaixin/retro_NP/terokit/rxn_cls_pro/all_rxnfps.pt',
                            save_folder='/amax/data/lishuaixin/retro_NP/terokit/rxn_cls_pro')
for i in range(5,15):
    kmeans_clustering_with_num_clusters(emb_folder='/amax/data/lishuaixin/retro_NP/terokit/rxn_cls_pro',
                                    num_clusters=i, save_folder='/amax/data/lishuaixin/retro_NP/terokit/rxn_cls_pro')

for folder_path in ['/amax/data/lishuaixin/retro_NP/terokit/teokit_complete_1.0/random_split',
                    '/amax/data/lishuaixin/retro_NP/terokit/teokit_complete_1.0/tanimoto_split/thres_0.6_frac_0.4',
                    '/amax/data/lishuaixin/retro_NP/terokit/teokit_complete_1.0/tanimoto_split/thres_0.4_frac_0.7']:
    for mode in ['test', 'val', 'train']:
        csv_path = f'{folder_path}/{mode}.csv'
        for dist in ['euc']:
            get_new_dataset(data_path=csv_path, k_cluster=6, distance=dist,
                rxn_cluster_path=f'/amax/data/lishuaixin/retro_NP/terokit/rxn_cls_pro/rxnfp_kmeans_{dist}/clusters_6/clustered_rxns.csv', only_pro=True)
            get_new_dataset(data_path=csv_path, k_cluster=8, distance=dist,
                rxn_cluster_path=f'/amax/data/lishuaixin/retro_NP/terokit/rxn_cls_pro/rxnfp_kmeans_{dist}/clusters_8/clustered_rxns.csv', only_pro=True)
            get_new_dataset(data_path=csv_path, k_cluster=10, distance=dist,
                rxn_cluster_path=f'//amax/data/lishuaixin/retro_NP/terokit/rxn_cls_pro/rxnfp_kmeans_{dist}/clusters_10/clustered_rxns.csv', only_pro=True)

'''
if __name__ == "__main__":

    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    pure_dataset_paths = ['/path/to/Terpene-former/data/TeroRXN/random_split',
                          '/path/to/Terpene-former/data/TeroRXN/tanimoto_split_thres0.6',
                          '/path/to/Terpene-former/data/TeroRXN/tanimoto_split_thres0.4']

    for pure_dataset_path in pure_dataset_paths:
        csv_path = f'{pure_dataset_path}/train_enz_rxnfp.csv'
        save_folder = f'{pure_dataset_path}/rxn_cls'
        if not os.path.exists(save_folder):
            # 如果保存文件夹不存在，则创建
            os.makedirs(save_folder, exist_ok=True)

        # 获取反应指纹和反应ID的pt文件
        all_rxnfps_pt_path, all_rxnids_pt_path = get_emb_pt(csv_path=csv_path, save_folder=save_folder, only_pro=False)
        
        # 对嵌入向量（embeddings）进行聚类分析，并通过绘图辅助选择最佳聚类数
        read_pt_files_and_cluster(emb_path=all_rxnfps_pt_path, save_folder=save_folder, save_fig=True)

        # 对嵌入向量进行KMeans聚类，并保存聚类结果
        num_cluster = 6
        _, rxn_cluster_path = kmeans_clustering_with_num_clusters(emb_folder=save_folder,
                                            distance='euc',
                                            num_clusters=num_cluster, 
                                            save_folder=save_folder)
        
        # 获取新的数据集，并将聚类结果添加到CSV文件中
        get_new_dataset(data_path=csv_path, k_cluster=num_cluster, distance='euc', 
                         rxn_cluster_path=rxn_cluster_path, only_pro=False)
        
        # 为测试集和验证集添加新的列
        for mode in ['test', 'val']:
            add_new_col_to_csv(in_path=f'{pure_dataset_path}/{mode}_enz_rxnfp.csv', 
                               out_path=f'{pure_dataset_path}/{mode}_enz_rxnfp.csv', 
                               col_name=f'rxnfp_{num_cluster}_euc', col_value=0)

        print(f'{pure_dataset_path} Done!')