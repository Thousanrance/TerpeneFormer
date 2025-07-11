from utils.build_utils import build_model, build_iterator, load_checkpoint, accumulate_batch
from utils.model_utils import validate, translate
from utils.loss_utils import LabelSmoothingLoss
from utils.smiles_utils import canonical_smiles
from utils.build_utils import fix_train_random_seed
from utils.check_utils import cal_torch_model_params


import re, json
import os
import copy
import math
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix
import numpy as np



parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda', help='device GPU/CPU')
parser.add_argument('--batch_size_trn', type=int, default=2, help='raw train batch size')
parser.add_argument('--batch_size_val', type=int, default=2, help='val/test batch size')
parser.add_argument('--batch_size_token', type=int, default=1048, help='train batch token number')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lr_scheduler', type=str, default='None', choices=['None', 'ReduceLROnPlateau'])
parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help='ReduceLROnPlateau factor')
parser.add_argument('--lr_scheduler_patience', type=int, default=10, help='ReduceLROnPlateau patience')
parser.add_argument('--min_lr', type=float, default=1e-5, help='minimum learning rate for ReduceLROnPlateau')

parser.add_argument('--log_dir', type=str, default='/path/to/log', help='log directory')
parser.add_argument('--data_dir', type=str, default='/path/to/TeroRXN/random_split', help='base directory')
parser.add_argument('--intermediate_dir', type=str, default='/path/to/intermediate', help='intermediate directory')
parser.add_argument('--checkpoint_dir', type=str, default='/path/to/ckpts', help='checkpoint directory')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint model file')
parser.add_argument('--encoder_num_layers', type=int, default=8, help='number of layers of transformer')
parser.add_argument('--decoder_num_layers', type=int, default=8, help='number of layers of transformer')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
parser.add_argument('--d_ff', type=int, default=2048, help='')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

# parser.add_argument('--known_class', type=str, default='True', choices=['True', 'False'],
#                     help='with reaction class known/unknown')
# parser.add_argument('--shared_vocab', type=str, default='False', choices=['True', 'False'], help='whether sharing vocab')
# parser.add_argument('--shared_encoder', type=str, default='False', choices=['True', 'False'],
#                     help='whether sharing encoder')
# parser.add_argument('--use_multi_modal', type=str, default='False', choices=['True', 'False'])
# parser.add_argument('--verbose', type=str, default='False', choices=['True', 'False'])

parser.add_argument('--known_class', action='store_true',
                    help='with reaction class known/unknown')
parser.add_argument('--shared_vocab', action='store_true', help='whether sharing vocab')
parser.add_argument('--shared_encoder', action='store_true',
                    help='whether sharing encoder')

parser.add_argument('--use_multi_modal', action='store_true')
parser.add_argument('--use_multi_modal_front', action='store_true', help='wether to add image modal in the front or not')
parser.add_argument('--use_multi_modal_after', action='store_true', help='concate picture dim in the hidden layer, \
                    and then use a FC layer to reduce dimention ')
parser.add_argument('--after_concate_layer', type=int, help='the layer of  concate picture dim in the hidden layer')
parser.add_argument('--after_big_imgdim', action='store_true',help='whther to choose big img dim 512 or not')
parser.add_argument('--image_default_size', type=int, default=224)
parser.add_argument('--residual_connect', action='store_true')
parser.add_argument('--cos_sim', action='store_true')
parser.add_argument('--attri_bondlen', action='store_true')

parser.add_argument('--new_trans', action='store_true')
parser.add_argument('--zero_init', action='store_true')
parser.add_argument('--sample', action='store_true')


parser.add_argument('--verbose', action='store_true')
parser.add_argument('--bond_atti_dim',type=int, default=7)
parser.add_argument('--USPTO_extend_vocab', action='store_true', )
parser.add_argument('--optim_loadstate', action='store_true')
parser.add_argument('--freeze_parms', action='store_true')

parser.add_argument('--max_epoch', type=int, default=1000, help='maximum epoch')
parser.add_argument('--max_step', type=int, default=600000, help='maximum steps')
parser.add_argument('--report_per_step', type=int, default=200, help='train loss reporting steps frequency')
parser.add_argument('--save_per_step', type=int, default=1000, help='checkpoint saving steps frequency')
parser.add_argument('--val_per_step', type=int, default=1000, help='validation steps frequency')

# enzymes
#parser.add_argument('--enz_token', action='store_true')
#parser.add_argument('--kmeans_label', type=str, default='seq', choices=['seq', 'esm'])
#parser.add_argument('--kmeans_num_clusters', type=int, default=10)

# rxnfp
parser.add_argument('--rxnfp_class', action='store_true')
parser.add_argument('--rxnfp_class_weights', type=float, nargs='+', default=None,
                    help='weight list for rxnfp classification loss')
#parser.add_argument('--rxnfp_token', action='store_true')
parser.add_argument('--rxnfp_label', type=str, default='rxnfp')
parser.add_argument('--rxnfp_dist', type=str, default='euc', choices=['euc','cos'])
parser.add_argument('--rxnfp_num_clusters', type=int, default=10)
parser.add_argument('--rxnfp_pro', action='store_true')

parser.add_argument('--model_origin', action='store_true')

parser.add_argument('--stepwise',action='store_true', help='')
parser.add_argument('--use_template', action='store_true', help='')
parser.add_argument('--beam_size', type=int, default=10, help='beam size')

args = parser.parse_args()


def anneal_prob(step, k=2, total=150000): # 随着步数的增多，概率趋近于1，也就是步数小时倾向于选择其它可能，步数增多时倾向稳定
    step = np.clip(step, 0, total)
    min_, max_ = 1, np.exp(k*1)
    return (np.exp(k*step/total) - min_) / (max_ - min_)


def main(args):
    fix_train_random_seed(42)

    summary = SummaryWriter(f'{args.checkpoint_dir}/tf')

    log_folder = args.log_dir
    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)
    # 时间日志 
    name_t = datetime.now().strftime("%D:%H:%M:%S").replace('/', ':') 
    log_file_name = log_folder + '/'+ name_t + '.txt'
    with open(log_file_name, 'a+') as f:
        f.write(str(args))
        f.write('\n')
    log_copy_file_name = args.checkpoint_dir +'/'+ name_t + '.txt'
    with open(log_copy_file_name, 'a+') as f:
        f.write(str(args))
        f.write('\n')

    print("Building iterator...")
    train_iter, val_iter, dataset_val, vocab_itos_src, vocab_itos_tgt = \
        build_iterator(args, train=True, sample=False, augment=True)
    print("Building iterator done.")
    print("Building model...")
    model = build_model(args, vocab_itos_src, vocab_itos_tgt)
    print("Building model done.")
    global_step = 1
    
    if args.checkpoint:
        print('Loading checkpoint...')
        global_step, optimizer_statedict, model = load_checkpoint(args, model)
        print('Loading checkpoint done.')
        global_step += 1

    print('globalstep', global_step)# 377502step，best model
    if args.freeze_parms:
        for name, param in model.named_parameters():
            if 'generator' not in name:
                param.requires_grad = False
            else:
                print('requires_grad',name)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.learning_rate, eps=1e-5)
    if args.optim_loadstate:
        optimizer.load_state_dict(optimizer_statedict)
    # print(cal_torch_model_params(model))
    # print('model structure', model)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_scheduler_factor, patience=args.lr_scheduler_patience, verbose=True, min_lr=args.min_lr)

    criterion_bond_rc = nn.BCELoss(reduction='sum')
    criterion_atom_rc = nn.BCELoss(reduction='sum')
    criterion_context_align = LabelSmoothingLoss(reduction='sum', smoothing=0.5)
    criterion_tokens = LabelSmoothingLoss(ignore_index=model.embedding_tgt.word_padding_idx,
                                          reduction='sum', apply_logsoftmax=False)
    weight_tensor = torch.tensor(args.rxnfp_class_weights, dtype=torch.float, device=args.device)
    criterion_rxnfp_class = nn.CrossEntropyLoss(weight=weight_tensor)

    loss_history_all, loss_history_token, loss_history_arc, loss_history_brc, loss_history_align, loss_history_rxnclass = [], [], [], [], [], []
    entry_count, src_max_length, tgt_max_length = 0, 0, 0
    final_src_len = 0

    true_batch = []

    if args.verbose == True:
        progress_bar = tqdm(train_iter)
    else:
        progress_bar = train_iter
    
    print('Training begin:')
    for epoch in range(args.max_epoch):
        print('epoch:', epoch)
        print('global_step:', global_step)
        if global_step > args.max_step:
            print('Finish training.')
            break
        # print('final_src_len', final_src_len)
        for batch in tqdm(progress_bar):
            if global_step > args.max_step:
                print('Finish training.')
                break

            model.train()
            raw_src, raw_tgt, rxnfp_class_label, *_ = batch
            # print('rxnfp_class_label.shape:', rxnfp_class_label.shape)
            # input('line 204 check:')
            # 不受到该batch的最大长度的影响
            if args.use_multi_modal and not args.use_multi_modal_after:
                src_max_length = 260
            else:
                src_max_length = max(src_max_length, raw_src.shape[0])
            tgt_max_length = max(tgt_max_length, raw_tgt.shape[0])
            entry_count += raw_tgt.shape[1]
            # print('src_max_length ',src_max_length)
            # final_src_len = max(src_max_length,final_src_len)
            if (src_max_length + tgt_max_length) * entry_count < args.batch_size_token: # 达不到设定的token长度的话，就进行累加
                
                true_batch.append(batch)
            else:
                # Accumulate Batch
                # print('true_batch',true_batch[0][0].shape)
                images = None
                src, tgt, rxnfp_class_label, gt_context_alignment, gt_nonreactive_mask, graph_packs, images = \
                        accumulate_batch(true_batch,use_multi_modal=args.use_multi_modal, args=args)
                bond, _ = graph_packs

                # print('src.shape:',src.shape)
                # print('rxnfp_class_label.shape:',rxnfp_class_label.shape)
                # input('line 227 check:')

                src, tgt, rxnfp_class_label, bond, gt_context_alignment, gt_nonreactive_mask = \
                    src.to(args.device), tgt.to(args.device), rxnfp_class_label.to(args.device), bond.to(args.device), \
                    gt_context_alignment.to(args.device), gt_nonreactive_mask.to(args.device)

                del true_batch
                torch.cuda.empty_cache()
                # print('src_itos',vocab_itos_src)
                # print('tgt_itos',vocab_itos_tgt)
                # print('tgt',tgt[:, 0])
                # gt = ''.join([vocab_itos_tgt[t] for t in tgt[:, 0] if t not in [1, 2, 3]])
                # print('gt',gt)
                # print('cano_gt',canonical_smiles(gt))

                p = np.random.rand()
                if p < anneal_prob(global_step): # 小于退火概率，不使用gt_nonreactive_mask
                    generative_scores, atom_rc_scores, bond_rc_scores, context_scores, rxnfp_class_scores = \
                        model(src, tgt, bond, None)
                else:
                    generative_scores, atom_rc_scores, bond_rc_scores, context_scores, rxnfp_class_scores = \
                        model(src, tgt, bond, gt_nonreactive_mask)
                
                # print('src.shape:',src.shape)
                # print('tgt.shape:',tgt.shape)
                # print('bond.shape:',bond.shape)
                # print('generative_scores.shape:',generative_scores.shape)
                # print('atom_rc_scores.shape:',atom_rc_scores.shape)
                # print('bond_rc_scores.shape:',bond_rc_scores.shape)
                # print('context_scores.shape:',context_scores[-1].shape)
                # print('rxnfp_class_scores.shape:',rxnfp_class_scores.shape)
                # input('line 258 check:')

                # Loss for language modeling:
                pred_token_logit = generative_scores.view(-1, generative_scores.size(2))
                gt_token_label = tgt[1:].view(-1)

                # print('pred_token_logit.shape:', pred_token_logit.shape)
                # print('gt_token_label.shape:', gt_token_label.shape)
                # input('line 266 check:')

                # Loss for atom-level reaction center identification:
                reaction_center_attn = ~gt_nonreactive_mask
                pred_atom_rc_prob = atom_rc_scores.view(-1)
                gt_atom_rc_label = reaction_center_attn.view(-1)

                # Loss for edge-level reaction center identification:
                if bond_rc_scores is not None:
                    pair_indices = torch.where(bond.sum(-1) > 0)
                    pred_bond_rc_prob = bond_rc_scores.view(-1)
                    gt_bond_rc_label = (reaction_center_attn[[pair_indices[1], pair_indices[0]]] & reaction_center_attn[
                        [pair_indices[2], pair_indices[0]]])
                    loss_bond_rc = criterion_bond_rc(pred_bond_rc_prob, gt_bond_rc_label.float())
                else:
                    loss_bond_rc = torch.zeros(1).to(src.device)

                # Loss for context alignment:
                is_inferred = (gt_context_alignment.sum(dim=-1) == 0)
                gt_context_align_label = gt_context_alignment[~is_inferred].view(-1, gt_context_alignment.shape[-1])

                # Compute all loss:
                loss_token = criterion_tokens(pred_token_logit, gt_token_label)
                loss_atom_rc = criterion_atom_rc(pred_atom_rc_prob, gt_atom_rc_label.float())
                loss_context_align = 0
                # for context_score in context_scores:
                context_score = context_scores[-1]
                pred_context_align_logit = context_score[~is_inferred].view(-1, context_score.shape[-1])

                loss_context_align += criterion_context_align(pred_context_align_logit,
                                                              gt_context_align_label) 

                # rxnfp分类损失
                # print('rxnfp_class_label:', rxnfp_class_label)
                # print('rxnfp_class_scores:', rxnfp_class_scores)
                # input('line 301 check:')
                loss_rxnfp_class = criterion_rxnfp_class(rxnfp_class_scores, rxnfp_class_label)
                    

                loss = loss_token + loss_atom_rc + loss_bond_rc + loss_context_align + loss_rxnfp_class

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history_all.append(loss.item())
                loss_history_token.append(loss_token.item())
                loss_history_arc.append(loss_atom_rc.item())
                loss_history_brc.append(loss_bond_rc.item())
                loss_history_align.append(loss_context_align.item())
                loss_history_rxnclass.append(loss_rxnfp_class.item())

                if global_step % args.report_per_step == 0:
                    if scheduler is not None:
                        current_lr = scheduler.optimizer.param_groups[0]['lr']
                        print(f'Step {global_step} learning rate: {current_lr:.10f}')

                    print_line = "[Epoch {} Iter {}] Loss {} NLL-Loss {} Rc-Loss {} {} Align-Loss {} Rxnclass-Loss {}".format(
                        epoch, global_step,
                        round(np.mean(loss_history_all), 4), round(np.mean(loss_history_token), 4),
                        round(np.mean(loss_history_arc), 4), round(np.mean(loss_history_brc), 4),
                        round(np.mean(loss_history_align), 4), round(np.mean(loss_history_rxnclass), 4))
                    print(print_line)
                    with open(log_file_name, 'a+') as f:
                        f.write(print_line)
                        f.write('\n')
                    with open(log_copy_file_name, 'a+') as f:
                        f.write(print_line)
                        f.write('\n')
                    summary.add_scalar('train/loss_all', np.mean(loss_history_all), global_step)
                    summary.add_scalar('train/NLL-Loss', np.mean(loss_history_token), global_step)
                    summary.add_scalar('train/Rc-Loss', np.mean(loss_history_arc), global_step)
                    summary.add_scalar('train/Align-Loss', np.mean(loss_history_brc), global_step)
                    summary.add_scalar('train/Rxnclass-Loss', np.mean(loss_history_rxnclass), global_step)


                    loss_history_all, loss_history_token, loss_history_arc, loss_history_brc, loss_history_align, loss_history_rxnclass = [], [], [], [], [], []

                if global_step % args.save_per_step == 0:
                    checkpoint_path = args.checkpoint_dir + '/model_{}.pt'.format(global_step)
                    torch.save({'model': model.state_dict(), 'step': global_step, 'optim': optimizer.state_dict()}, checkpoint_path)
                    print('Checkpoint saved to {}'.format(checkpoint_path))
                    if global_step > 0:
                        # input('line 345 check-translate will begin:')
                        products, ground_truths, generations, rxn_class_labels, predicted_rxn_class_labels = translate(val_iter, model, dataset_val, args=args)
                        # print('products:', type(products), len(products))
                        # print('ground_truths:', type(ground_truths), len(ground_truths))
                        # print('generations:', type(generations), len(generations))
                        # print('rxn_class_labels:', type(rxn_class_labels), len(rxn_class_labels))
                        # print('predicted_rxn_class_labels:', type(predicted_rxn_class_labels), len(predicted_rxn_class_labels))
                        # print('rxn_class_labels:',rxn_class_labels)
                        # print('predicted_rxn_class_labels:',predicted_rxn_class_labels)
                        # input('line 354 check:')
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
                        
                        # print('cano_pros:', len(cano_pros))
                        # print('cano_gts:', len(cano_gts))
                        # print('rxn_class_labels:', len(rxn_class_labels))

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
                        
                        csv_path = args.checkpoint_dir + f'/val_dataset_{global_step}_TOP{args.beam_size}.csv'
                        df.to_csv(csv_path, index=False)

                        print(f'model_{global_step}:')
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
                        
                        acc_score_csv_path = args.checkpoint_dir + '/val_acc_score.csv'
                        if os.path.exists(acc_score_csv_path):
                            acc_score_df = pd.read_csv(acc_score_csv_path)
                        else:
                            acc_score_df = pd.DataFrame(columns=['steps', 'acc_score', 'class_acc'])
                        acc_dict = {'steps': global_step, 'acc_score': acc_score, 'class_acc': class_acc}
                        acc_score_df = acc_score_df.append(acc_dict, ignore_index=True)
                        acc_score_df.to_csv(acc_score_csv_path, index=False)

                        if scheduler is not None:
                            scheduler.step(acc_score)

                if global_step % args.val_per_step == 0:
                    if scheduler is not None:
                        current_lr = scheduler.optimizer.param_groups[0]['lr']
                        print(f'Step {global_step} learning rate: {current_lr:.10f}')    

                    accuracy_arc, accuracy_brc, accuracy_token, accuracy_rxnclass = \
                        validate(model, val_iter, args, model.embedding_tgt.word_padding_idx)
                    print_line = 'Validation accuracy: {} - {} - {} - {}'.format(round(accuracy_arc, 4),
                                                                            round(accuracy_brc, 4),
                                                                            round(accuracy_token, 4), 
                                                                            round(accuracy_rxnclass, 4))
                        
                    print(print_line)
                    summary.add_scalar('train/accuracy_arc', accuracy_arc, global_step)
                    summary.add_scalar('train/accuracy_brc', accuracy_brc, global_step)
                    summary.add_scalar('train/accuracy_token', accuracy_token, global_step)
                    summary.add_scalar('train/accuracy_rxnclass', accuracy_rxnclass, global_step)

                    with open(log_file_name, 'a+') as f:
                        f.write(print_line)
                        f.write('\n')
                    
                    with open(log_copy_file_name, 'a+') as f:
                        f.write(print_line)
                        f.write('\n')

                # Restart Accumulation
                global_step += 1
                true_batch = [batch]
                entry_count, src_max_length, tgt_max_length = raw_src.shape[1], raw_src.shape[0], raw_tgt.shape[0]
    print('Training done.')

if __name__ == '__main__':
    print(args)
    with open('args.pk', 'wb') as f:
        pickle.dump(args, f)

    if args.known_class:
        args.checkpoint_dir = args.checkpoint_dir + '_typed'
    else:
        args.checkpoint_dir = args.checkpoint_dir + '_untyped'
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    main(args)
