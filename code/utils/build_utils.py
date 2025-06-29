from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


from functools import partial
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import RetroDataset, Multi_Step_RetroDataset
from models.model import RetroModel

import shutil  
import os  

  
def copy_and_rename_file(source_file_path, target_file_path):  
  
    if not os.path.exists(source_file_path):  
        print(f"Source {source_file_path} does not exist")  
        return  
  
    # Copy the file  
    try:  
        shutil.copy2(source_file_path, target_file_path)  
        print(f"File {source_file_path} has been copied to {target_file_path}")  
    except Exception as e:  
        print(f"Error occurred while copying the file: {e}")

def fix_train_random_seed(seed=42):
    # fix random seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def load_checkpoint(args, model):
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        checkpoint_path = args.checkpoint
    print('Loading checkpoint from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['model'])
    module_items = {k.replace('module.',''):v for k,v in checkpoint['model'].items()}
    # print('module_name.keys()',module_name.keys())
    if args.USPTO_extend_vocab:
        for key in ['embedding_src.token.weight', 'embedding_src.position.pe', 'embedding_tgt.token.weight', \
                    'embedding_tgt.position.pe', 'encoder.embeddings.token.weight', 'encoder.embeddings.position.pe', \
                    'decoder.embeddings.token.weight', 'decoder.embeddings.position.pe',\
                    'generator.0.weight', 'generator.0.bias']:
            # module_items = module_items.pop(key)
            del module_items[key]

    # 过滤掉模型中没有的参数（比如新加的rxn_classifier）
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in module_items.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(filtered_dict)

    model.load_state_dict(model_dict, False) # 去掉DataParallel 预训练model中的module(可行权重值一致):
    optimizer = checkpoint['optim']
    step = checkpoint['step']
    step += 1
    return step, optimizer, model.to(args.device)


def build_model(args, vocab_itos_src, vocab_itos_tgt):
    src_pad_idx = np.argwhere(np.array(vocab_itos_src) == '<pad>')[0][0]
    tgt_pad_idx = np.argwhere(np.array(vocab_itos_tgt) == '<pad>')[0][0]

    model = RetroModel(
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        d_model=args.d_model, heads=args.heads, d_ff=args.d_ff, dropout=args.dropout,
        vocab_size_src=len(vocab_itos_src), vocab_size_tgt=len(vocab_itos_tgt),
        shared_vocab=args.shared_vocab, shared_encoder=args.shared_encoder,
        src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx,
        args=args)

    return model.to(args.device)


def build_iterator(args, train=True, sample=False, augment=False):
    
    if train:
        dataset = RetroDataset(mode='train', data_folder=args.data_dir,
                               intermediate_folder=args.intermediate_dir,
                               known_class=args.known_class,
                               shared_vocab=args.shared_vocab, sample=sample, augment=augment,
                               use_multi_modal = args.use_multi_modal,
                                args=args)
        dataset_val = RetroDataset(mode='val', data_folder=args.data_dir,
                                   intermediate_folder=args.intermediate_dir,
                                   known_class=args.known_class,
                                   shared_vocab=args.shared_vocab, sample=sample,
                                   use_multi_modal = args.use_multi_modal,
                                   args=args)
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        train_iter = DataLoader(dataset, batch_size=args.batch_size_trn, shuffle=not sample,  # num_workers=8,
                                collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device,
                                                   use_multi_modal = args.use_multi_modal, args=args))
        val_iter = DataLoader(dataset_val, batch_size=args.batch_size_val, shuffle=False,  # num_workers=8,
                              collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device,
                                                 use_multi_modal = args.use_multi_modal, args=args))
        return train_iter, val_iter, dataset_val, dataset.src_itos, dataset.tgt_itos

    else:
        if args.multi_step_pred:
            dataset = Multi_Step_RetroDataset(mode='test', 
                    known_class=args.known_class,
                    shared_vocab=args.shared_vocab,
                    args=args)
        else:
            if args.get_val_result:
                mode = 'val'
            else:
                mode = 'test'
            dataset = RetroDataset(mode=mode, data_folder=args.data_dir,
                                intermediate_folder=args.intermediate_dir,
                                known_class=args.known_class,
                                shared_vocab=args.shared_vocab,
                                use_multi_modal = args.use_multi_modal,
                                sample=args.sample,
                                args=args)
        src_pad, tgt_pad = dataset.src_stoi['<pad>'], dataset.tgt_stoi['<pad>']
        test_iter = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=False,  # num_workers=8,
                               collate_fn=partial(collate_fn, src_pad=src_pad, tgt_pad=tgt_pad, device=args.device,
                                                  use_multi_modal = args.use_multi_modal, args=args))
        return test_iter, dataset


def collate_fn(data, src_pad, tgt_pad, device='cuda',use_multi_modal=False, args=None):
    # collate_fn会改变数据的堆叠方式
    """Build mini-batch tensors:
    :param sep: (int) index of src seperator
    :param pads: (tuple) index of src and tgt padding
    """

    if use_multi_modal:
        
        src, src_graph, tgt, alignment, nonreactive_mask, rxnfp_class_label, image = zip(*data)
    else:
        # Sort a data list by caption length
        src, src_graph, tgt, alignment, nonreactive_mask, rxnfp_class_label = zip(*data)
    max_src_length = max([len(s) for s in src])
    max_tgt_length = max([len(t) for t in tgt])

    anchor = torch.zeros([], device=device)

    # Graph structure with edge attributes
    if args.bond_atti_dim == 7:
        new_bond_matrix = anchor.new_zeros((len(data), max_src_length, max_src_length, 7), dtype=torch.long)
    else:
        new_bond_matrix = anchor.new_zeros((len(data), max_src_length, max_src_length, args.bond_atti_dim), dtype=torch.float)
    # Pad_sequence
    new_src = anchor.new_full((max_src_length, len(data)), src_pad, dtype=torch.long)
    new_tgt = anchor.new_full((max_tgt_length, len(data)), tgt_pad, dtype=torch.long)
    new_alignment = anchor.new_zeros((len(data), max_tgt_length - 1, max_src_length), dtype=torch.float)
    new_nonreactive_mask = anchor.new_ones((max_src_length, len(data)), dtype=torch.bool)

    for i in range(len(data)):
        new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])
        new_nonreactive_mask[:, i][:len(nonreactive_mask[i])] = torch.BoolTensor(nonreactive_mask[i])
        new_tgt[:, i][:len(tgt[i])] = torch.LongTensor(tgt[i])
        new_alignment[i, :alignment[i].shape[0], :alignment[i].shape[1]] = alignment[i].float()

        full_adj_matrix = torch.from_numpy(src_graph[i].full_adjacency_tensor)
        new_bond_matrix[i, 1:full_adj_matrix.shape[0]+1, 1:full_adj_matrix.shape[1]+1] = full_adj_matrix

    # 将rxnfp_class_label转为tensor
    rxnfp_class_label = torch.LongTensor(rxnfp_class_label)

    if use_multi_modal:
        return new_src, new_tgt, rxnfp_class_label, new_alignment, new_nonreactive_mask, (new_bond_matrix, src_graph), image
    else:
        return new_src, new_tgt, rxnfp_class_label, new_alignment, new_nonreactive_mask, (new_bond_matrix, src_graph)


def accumulate_batch(true_batch, src_pad=1, tgt_pad=1, use_multi_modal=False, args=None):
    """
    src, tgt, rxn_class_label, context_alignment, nonreactive_mask, (bond, src_graph), image = true_batch[i]
    其中src_graph 为SmilesGraph类
    
    """
    # print('accumulate_batch',args)
    src_max_length, tgt_max_length, entry_count = 0, 0, 0
    batch_size = true_batch[0][0].shape[1]

    if use_multi_modal:
        for batch in true_batch:
            src, tgt, rxn_class_label, *_ = batch
            if args.use_multi_modal_after:
                src_max_length = max(src.shape[0], src_max_length)
            else:
                src_max_length = 280
            tgt_max_length = max(tgt.shape[0], tgt_max_length)
            entry_count += tgt.shape[1]
    else:
        for batch in true_batch:
            src, tgt, rxnfp_class_label, *_ = batch
            src_max_length = max(src.shape[0], src_max_length)
            tgt_max_length = max(tgt.shape[0], tgt_max_length)
            entry_count += tgt.shape[1]
    new_src = torch.zeros((src_max_length, entry_count)).fill_(src_pad).long()
    new_tgt = torch.zeros((tgt_max_length, entry_count)).fill_(tgt_pad).long()
    # print('new_src',new_src.size())
    # print('new_tgt',new_tgt.size())
    new_context_alignment = torch.zeros((entry_count, tgt_max_length - 1, src_max_length)).float()
    new_nonreactive_mask = torch.ones((src_max_length, entry_count)).bool()

    # Graph packs:
    if args.bond_atti_dim == 7:
        new_bond_matrix = torch.zeros((entry_count, src_max_length, src_max_length, 7)).long()
    else:
        new_bond_matrix = torch.zeros((entry_count, src_max_length, src_max_length, args.bond_atti_dim)).float()
    new_src_graph_list = []
    
    new_image=None
    if use_multi_modal:
        # default single image size (3,184,184)
        img_size = true_batch[0][5][0].shape
        len_ch, len_w, len_h = img_size
        new_image = torch.zeros((entry_count, len_ch, len_w, len_h)).float()
    left = 0
    right = 0

    all_rxnfp_class_label = []

    if use_multi_modal:
        for i in range(len(true_batch)):
            src, tgt, rxnfp_class_label, context_alignment, nonreactive_mask, graph_packs, image = true_batch[i]
            # print('image',len(image))
            # print('single image', image[0].shape)
            image = torch.stack(image)
            # print('single image', image.shape)
            # print('src',src.size())
            # print('tgt',tgt.size())
            bond, src_graph = graph_packs 
            right += src.shape[1]
            # new_src[:, batch_size * i: batch_size * (i + 1)][:src.shape[0]] = src
            # new_nonreactive_mask[:, batch_size * i: batch_size * (i + 1)][:nonreactive_mask.shape[0]] = nonreactive_mask
            # new_tgt[:, batch_size * i: batch_size * (i + 1)][:tgt.shape[0]] = tgt
            # new_context_alignment[batch_size * i: batch_size * (i + 1), :context_alignment.shape[1], :context_alignment.shape[2]] = context_alignment
            # new_bond_matrix[batch_size * i: batch_size * (i + 1), :bond.shape[1], :bond.shape[2]] = bond
            new_src[:, left: right][:src.shape[0]] = src
            new_nonreactive_mask[:, left: right][:nonreactive_mask.shape[0]] = nonreactive_mask
            new_tgt[:, left: right][:tgt.shape[0]] = tgt
            new_context_alignment[left: right, :context_alignment.shape[1], :context_alignment.shape[2]] = context_alignment
            new_bond_matrix[left: right, :bond.shape[1], :bond.shape[2]] = bond
            new_image[left: right, :, :, :] = image

            new_src_graph_list += src_graph
            left = right

            all_rxnfp_class_label.append(rxnfp_class_label)

    else:
        for i in range(len(true_batch)):
            src, tgt, rxnfp_class_label, context_alignment, nonreactive_mask, graph_packs = true_batch[i]
            # print('src',src.size())
            # print('tgt',tgt.size())
            bond, src_graph = graph_packs
            right += src.shape[1]
            # new_src[:, batch_size * i: batch_size * (i + 1)][:src.shape[0]] = src
            # new_nonreactive_mask[:, batch_size * i: batch_size * (i + 1)][:nonreactive_mask.shape[0]] = nonreactive_mask
            # new_tgt[:, batch_size * i: batch_size * (i + 1)][:tgt.shape[0]] = tgt
            # new_context_alignment[batch_size * i: batch_size * (i + 1), :context_alignment.shape[1], :context_alignment.shape[2]] = context_alignment
            # new_bond_matrix[batch_size * i: batch_size * (i + 1), :bond.shape[1], :bond.shape[2]] = bond
            new_src[:, left: right][:src.shape[0]] = src
            new_nonreactive_mask[:, left: right][:nonreactive_mask.shape[0]] = nonreactive_mask
            new_tgt[:, left: right][:tgt.shape[0]] = tgt
            new_context_alignment[left: right, :context_alignment.shape[1], :context_alignment.shape[2]] = context_alignment
            new_bond_matrix[left: right, :bond.shape[1], :bond.shape[2]] = bond
            new_src_graph_list += src_graph
            left = right

            all_rxnfp_class_label.append(rxnfp_class_label)

    if isinstance(all_rxnfp_class_label[0], torch.Tensor):
        new_rxnfp_class_label = torch.cat(all_rxnfp_class_label, dim=0)
    else:
        new_rxnfp_class_label = np.concatenate(all_rxnfp_class_label, axis=0)

    return new_src, new_tgt, new_rxnfp_class_label, new_context_alignment, new_nonreactive_mask, \
            (new_bond_matrix, new_src_graph_list), new_image

