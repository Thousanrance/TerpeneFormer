import copy
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from utils.build_utils import accumulate_batch

from utils.smiles_utils import *
from utils.translate_utils import translate_batch_original, translate_batch_stepwise

def reallocate_batch(batch, location='cpu'):
    batch = list(batch)
    for i in range(len(batch)):
        batch[i] = batch[i].to(location)
    return tuple(batch)


def validate(model, val_iter, args=None, pad_idx=1):
    pred_token_list, gt_token_list, pred_infer_list, gt_infer_list = [], [], [], []
    pred_arc_list, gt_arc_list = [], []
    pred_brc_list, gt_brc_list = [], []
    all_pred_rxn_class = []
    all_gt_rxn_class = []
    model.eval()
    for batch in tqdm(val_iter):
        true_batch = []
        true_batch.append(batch)

        src, tgt, rxn_class_label, gt_context_alignment, gt_nonreactive_mask, graph_packs, _ = \
            accumulate_batch(true_batch, use_multi_modal=args.use_multi_modal, args=args)
        bond, _ = graph_packs
        src, tgt, rxn_class_label, bond, gt_context_alignment, gt_nonreactive_mask = \
            src.to(args.device), tgt.to(args.device), rxn_class_label.to(args.device), bond.to(args.device), \
            gt_context_alignment.to(args.device), gt_nonreactive_mask.to(args.device)


        # Infer:
        with torch.no_grad():
            scores, atom_rc_scores, bond_rc_scores, context_alignment, rxn_class_scores = \
                model(src, tgt, bond)
            context_alignment = F.softmax(context_alignment[-1], dim=-1)

        # Atom-level reaction center accuracy:
        pred_arc = (atom_rc_scores.squeeze(2) > 0.5).bool()
        pred_arc_list += list(~pred_arc.view(-1).cpu().numpy())
        gt_arc_list += list(gt_nonreactive_mask.view(-1).cpu().numpy())

        # Bond-level reaction center accuracy:
        if bond_rc_scores is not None:
            pred_brc = (bond_rc_scores > 0.5).bool()
            pred_brc_list += list(pred_brc.view(-1).cpu().numpy())

        pair_indices = torch.where(bond.sum(-1) > 0)
        rc = ~gt_nonreactive_mask
        gt_bond_rc_label = (rc[[pair_indices[1], pair_indices[0]]] & rc[[pair_indices[2], pair_indices[0]]])
        gt_brc_list += list(gt_bond_rc_label.view(-1).cpu().numpy())

        # Token accuracy:
        pred_token_logit = scores.view(-1, scores.size(2))
        _, pred_token_label = pred_token_logit.topk(1, dim=-1)
        gt_token_label = tgt[1:].view(-1)
        pred_token_list.append(pred_token_label[gt_token_label != pad_idx])
        gt_token_list.append(gt_token_label[gt_token_label != pad_idx])

        # 记录所有batch的类别预测和真实标签
        all_pred_rxn_class.append(rxn_class_scores.argmax(dim=-1).cpu())
        all_gt_rxn_class.append(rxn_class_label.cpu())

    pred_tokens = torch.cat(pred_token_list).view(-1)
    gt_tokens = torch.cat(gt_token_list).view(-1)

    # 合并所有batch的类别预测和真实标签
    all_pred_rxn_class = torch.cat(all_pred_rxn_class)
    all_gt_rxn_class = torch.cat(all_gt_rxn_class)
    rxn_class_acc = (all_pred_rxn_class == all_gt_rxn_class).float().mean().item()

    if bond_rc_scores is not None:
        return np.mean(np.array(pred_arc_list) == np.array(gt_arc_list)), \
               np.mean(np.array(pred_brc_list) == np.array(gt_brc_list)), \
               (pred_tokens == gt_tokens).float().mean().item(), \
                rxn_class_acc
    else:
        return np.mean(np.array(pred_arc_list) == np.array(gt_arc_list)), \
               0, \
               (pred_tokens == gt_tokens).float().mean().item(), \
                rxn_class_acc


def translate(iterator, model, dataset, args):
    ground_truths = []
    products = []
    generations = []
    predicted_rxn_class_labels = []
    rxn_class_labels = []
    invalid_token_indices = [dataset.tgt_stoi['<RX_{}>'.format(i)] for i in range(1, 11)]
    invalid_token_indices += [dataset.tgt_stoi['<UNK>'], dataset.tgt_stoi['<unk>']]
    # Translate:
    for batch in tqdm(iterator, total=len(iterator)):
        src, tgt, rxn_class_label, *_ = batch
        rxn_class_labels.extend(rxn_class_label.cpu().numpy())

        if not args.stepwise:
            # Original Main:
            pred_tokens, pred_scores, predicted_rxn_class_label = translate_batch_original(model, batch, beam_size=args.beam_size,
                                                                invalid_token_indices=invalid_token_indices, args=args)
            predicted_rxn_class_labels.extend(predicted_rxn_class_label) # predicted_rxn_class_label是一个numpy
            for idx in range(batch[0].shape[1]): # batch[0] 相当于src，然后src形状是[len,bs]
                # print('itos',dataset.get_itos())
                # print('tgt',tgt[:, idx])
                # print('batch[0]',batch[0].shape)
                pro = ''.join(dataset.reconstruct_smi(src[:, idx], src=True))
                gt = ''.join(dataset.reconstruct_smi(tgt[:, idx], src=False))
                # print('gt',gt)
                # print('cano_gt',canonical_smiles(gt))
                hypos = np.array([''.join(dataset.reconstruct_smi(tokens, src=False)) for tokens in pred_tokens[idx]])
                hypo_len = np.array([len(smi_tokenizer(ht)) for ht in hypos])
                new_pred_score = copy.deepcopy(pred_scores[idx]).cpu().numpy() / hypo_len
                ordering = np.argsort(new_pred_score)[::-1]

                products.append(pro)
                ground_truths.append(gt)
                generations.append(hypos[ordering])
        else: # 用不到，没改
            # Stepwise Main:
            # untyped: T=10; beta=0.5, percent_aa=40, percent_ab=40
            # typed: T=10; beta=0.5, percent_aa=40, percent_ab=55
            if args.known_class:
                percent_ab = 55
            else:
                percent_ab = 40
            pred_tokens, pred_scores, predicts = \
                translate_batch_stepwise(model, batch, beam_size=args.beam_size,
                                         invalid_token_indices=invalid_token_indices,
                                         T=10, alpha_atom=-1, alpha_bond=-1,
                                         beta=0.5, percent_aa=40, percent_ab=percent_ab, k=3,
                                         use_template=args.use_template,
                                         factor_func=dataset.factor_func,
                                         reconstruct_func=dataset.reconstruct_smi,
                                         rc_path=args.intermediate_dir + '/rt2reaction_center.pk')

            original_beam_size = pred_tokens.shape[1]
            current_i = 0
            for batch_i, predict in enumerate(predicts):
                gt = ''.join(dataset.reconstruct_smi(tgt[:, batch_i], src=False))
                remain = original_beam_size
                beam_size = math.ceil(original_beam_size / len(predict))

                # normalized_reaction_center_score = np.array([pred[1] for pred in predict]) / 10
                hypo_i, hypo_scores_i = [], []
                for j, (rc, rc_score) in enumerate(predict):
                    # rc_score = normalized_reaction_center_score[j]

                    pred_token = pred_tokens[current_i + j]

                    sub_hypo_candidates, sub_score_candidates = [], []
                    for k in range(pred_token.shape[0]):
                        hypo_smiles_k = ''.join(dataset.reconstruct_smi(pred_token[k], src=False))
                        hypo_lens_k = len(smi_tokenizer(hypo_smiles_k))
                        hypo_scores_k = pred_scores[current_i + j][k].cpu().numpy() / hypo_lens_k + rc_score / 10

                        if hypo_smiles_k not in hypo_i:  # only select unique entries
                            sub_hypo_candidates.append(hypo_smiles_k)
                            sub_score_candidates.append(hypo_scores_k)

                    ordering = np.argsort(sub_score_candidates)[::-1]
                    sub_hypo_candidates = list(np.array(sub_hypo_candidates)[ordering])[:min(beam_size, remain)]
                    sub_score_candidates = list(np.array(sub_score_candidates)[ordering])[:min(beam_size, remain)]

                    hypo_i += sub_hypo_candidates
                    hypo_scores_i += sub_score_candidates

                    remain -= beam_size

                current_i += len(predict)
                ordering = np.argsort(hypo_scores_i)[::-1][:args.beam_size]
                ground_truths.append(gt)
                generations.append(np.array(hypo_i)[ordering])
        # break
    return  products, ground_truths, generations, rxn_class_labels, predicted_rxn_class_labels


