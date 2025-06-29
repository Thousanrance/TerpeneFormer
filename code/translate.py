from utils.smiles_utils import *
# from retroformer.utils.translate_utils import translate_batch_original, translate_batch_stepwise
from utils.build_utils import build_model, build_iterator, load_checkpoint, copy_and_rename_file
from utils.parsing_utils import get_translate_args
from utils.model_utils import translate
import re, sys
import os, json
import copy
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix

def main(args):
    # Build Data Iterator:
    print('Building data iterator...')
    iterator, dataset = build_iterator(args, train=False)
    print('Building data iterator done!')

    # Load Checkpoint Model:
    print('Loading checkpoint model...')
    model = build_model(args, dataset.src_itos, dataset.tgt_itos)
    _, _, model = load_checkpoint(args, model)
    print('Loading checkpoint model done!')

    # Get Output Path:
    dec_version = 'stepwise' if args.stepwise else 'vanilla'
    exp_version = 'typed' if args.known_class else 'untyped'
    aug_version = '_augment' if 'augment' in args.checkpoint_dir else ''
    tpl_version = '_template' if args.use_template else ''
    pt_name = args.checkpoint.split('.')[0]
    # file_name = 'result/{}_{}_bs_top{}_generation_{}{}{}.pk'.format(pt_name,dec_version, args.beam_size, exp_version,
    #                                                                 aug_version, tpl_version)

    # output_path = os.path.join(args.intermediate_dir, file_name)
    # print('Output path: {}'.format(output_path))
    # file_dir = os.path.dirname(output_path)
    # os.makedirs(file_dir,exist_ok=True)
    # Begin Translating:
    print('Translating...')
    products, ground_truths, generations, rxn_class_labels, predicted_rxn_class_labels = translate(iterator, model, dataset, args)

    cano_pros = [canonical_smiles(pro) for pro in products]
    cano_gts = [canonical_smiles(gt) for gt in ground_truths]
    accuracy_matrix = np.zeros((len(ground_truths), args.beam_size)) # 前j个预测底物里是否有对的
    equal_matrix = np.zeros((len(ground_truths), args.beam_size)) # 每个预测底物是否是对的
    for i in range(len(ground_truths)):
        gt_i = canonical_smiles(ground_truths[i])
        generation_i = [canonical_smiles(gen) for gen in generations[i]]
        for j in range(args.beam_size):
            if gt_i in generation_i[:j + 1]:
                accuracy_matrix[i][j] = 1
            if gt_i == generation_i[j]:
                equal_matrix[i][j] = 1

    df = pd.DataFrame({'pro':cano_pros, 'gt_sub':cano_gts, 
                        'rxnfp_class': rxn_class_labels, 'pred_rxnfp_class': predicted_rxn_class_labels, 
                        'class_judge': None, 'judge':None, 'correct_count': None,  
                        'pred_1':None,'pred_2':None,'pred_3':None,'pred_4':None,'pred_5':None,
                        'pred_6':None, 'pred_7':None,'pred_8':None,'pred_9':None,'pred_10':None})
                        
    for idx, row in df.iterrows():
        df.at[idx, 'correct_count'] = np.sum(equal_matrix[idx][:])
        if rxn_class_labels[idx] == predicted_rxn_class_labels[idx]:
            df.at[idx, 'class_judge'] = True
        else:
            df.at[idx, 'class_judge'] = False
        if np.sum(accuracy_matrix[idx][:])==0:
            df.at[idx, 'judge'] = False
        else:
            df.at[idx, 'judge'] = True
        for j in range(10):
            df.at[idx, f'pred_{j+1}'] = generations[idx][j]
        
    # 统计class_judge为True/False时，correct_count的平均个数
    true_mask = df['class_judge'] == True
    false_mask = df['class_judge'] != True

    true_mean = df.loc[true_mask, 'correct_count'].astype(float).mean()
    false_mean = df.loc[false_mask, 'correct_count'].astype(float).mean()
    correct_count_mean = df['correct_count'].astype(float).mean()

    print(f"Average correct_count: {correct_count_mean:.3f}")
    print(f"Average correct_count when class_judge=True: {true_mean:.3f}")
    print(f"Average correct_count when class_judge=False: {false_mean:.3f}")
    

    val_ = 'val_' if args.get_val_result else ''

    if '/' not in args.checkpoint:
        csv_path = args.checkpoint_dir + f'/{val_}predictions_{args.checkpoint}TOP{args.beam_size}.csv'
    else:
        csv_path = args.checkpoint_dir + f'/{val_}predictions_TOP{args.beam_size}.csv'
    df.to_csv(csv_path, index=False)
    # with open(output_path, 'wb') as f:
    #     pickle.dump((ground_truths, generations), f)

    acc_score = 0.
    for j in range(args.beam_size):
        acc_j = round(np.mean(accuracy_matrix[:, j]), 3)
        print('Top-{}: {}'.format(j + 1, acc_j))
        if j+1 ==1:
            acc_score += 10*acc_j
        elif j+1 ==3:
            acc_score += 8*acc_j
        elif j+1 ==5:
            acc_score += 6*acc_j
        elif j+1 ==10:
            acc_score += 1*acc_j
        else:
            pass
    
    class_acc = np.mean(np.array(rxn_class_labels) == np.array(predicted_rxn_class_labels))

    acc_score_csv_path = args.checkpoint_dir + '/test_acc_score.csv'
    if args.get_val_result:
        acc_score_csv_path = args.checkpoint_dir + '/val_acc_score.csv'
    
    if os.path.exists(acc_score_csv_path):
        acc_score_df = pd.read_csv(acc_score_csv_path)
    else:
        acc_score_df = pd.DataFrame(columns=['steps', 'acc_score', 'class_acc'])
    acc_dict = {'steps':args.steps, 'acc_score':acc_score, 'class_acc':class_acc}
    acc_score_df = acc_score_df.append(acc_dict, ignore_index=True)
    acc_score_df.to_csv(acc_score_csv_path, index=False)

    print('Translation done!')

    return


if __name__ == "__main__":
    # if args.known_class:
    #     args.checkpoint_dir = args.checkpoint_dir + '_typed'
    # else:
    #     args.checkpoint_dir = args.checkpoint_dir + '_untyped'
    args = get_translate_args()
    if not args.get_val_result:
        sys.stdout = open(os.path.join(args.checkpoint_dir, 'translate.log'), 'a+')   
        print(args)
        
    if args.use_template:
        args.stepwise = True
    
    if args.ckpt_range:
        steps_list = [i for i in range(args.range_end, args.range_begin,-args.range_step)]
        for steps in steps_list:
            args.steps = steps
            args.checkpoint = f'model_{steps}.pt'
            checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
            if not os.path.exists(checkpoint_path):
                continue
            main(args)
    elif args.ckpt_re_pt:
        for filename in os.listdir(args.checkpoint_dir):
            # Check if the file ends with ".pt"
            if filename.endswith('.pt'):
                # Use regular expression to match the numbers in the filename
                match = re.search(r'\d+', filename)
                if match:
                    steps = int(match.group())
                    args.steps = steps
                    args.checkpoint = f'model_{steps}.pt'
                    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
                    if not os.path.exists(checkpoint_path):
                        continue
                    main(args)
    else:
        args.steps = None
        main(args)

    # acc_score_csv_path = args.checkpoint_dir + '/test_acc_score.csv'
    # acc_score_df = pd.read_csv(acc_score_csv_path)
    # acc_score_df.sort_values(by='acc_score', inplace=True, ascending=True)
    # max_acc_score_row = acc_score_df.iloc[-1]
    # print('max_acc_score_dict',max_acc_score_row)

    # checkpoint_path = args.checkpoint_dir + '/model_{}.pt'.format(max_acc_score_row['steps'])
    # save_ckpt_path = args.checkpoint_dir + '/model_best.pt'
    if not args.get_val_result:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    print('All translation done!')
    # copy_and_rename_file(checkpoint_path,save_ckpt_path)

