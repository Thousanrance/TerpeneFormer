import pickle
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import Chem
from utils.build_utils import build_model, load_checkpoint
from copy import deepcopy
import torch
def check_vocab(vocab_path='/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate/thres_0.6_frac_0.4/vocab_share.pk'):
    with open(vocab_path, 'rb') as f:
        src_itos, tgt_itos = pickle.load(f)
    print('src_itos', src_itos)
    print(type(src_itos))
    print('tgt_itos', tgt_itos)
    print(len(src_itos))

def remove_repeated_tokens(ori_list, add_list):
    ori_list_clone = deepcopy(ori_list)
    for token in add_list:
        if token not in ori_list_clone:
            ori_list_clone.append(token)
    return ori_list_clone
    
def get_USTPO_finetune_vocab(ori_vocab_path='/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate',
                              update_vocab_path='/home/lishuaixin/lsx/my/retro_NP/baseline/Retroformer/retroformer/intermediate/finetuned',
                              shared_vocab=True):
    '''to get USTPO finetune vocab, we need to merge the two sets:  USTPO vocab sets and terokit vocab sets

    '''
    # FIXME 这个由于字典原因，需要先产生共用的字典，然后从头训练得到USPTO的ckpts，然后（这时我们必须要求所有的finetune字典相同，才能避免用三个字典训练USPTO50K数据）
    # 只需要得到一个字典就可以
    
    
    with open(ori_vocab_path+'/USPTO_50K/vocab_share.pk', 'rb') as f:
        USPTO_src_itos, USPTO_tgt_itos = pickle.load(f)
    with open(ori_vocab_path+'/random_split/vocab_share.pk', 'rb') as f:
        tero_rand_src_itos, tero_rand_tgt_itos = pickle.load(f)
    with open(ori_vocab_path+'/thres_0.6_frac_0.4/vocab_share.pk', 'rb') as f:
        tero_thre64_src_itos, tero_thre64_tgt_itos = pickle.load(f)
    with open(ori_vocab_path+'/thres_0.4_frac_0.7/vocab_share.pk', 'rb') as f:
        tero_thre47_src_itos, tero_thre47_tgt_itos = pickle.load(f)

    finetune_rand_src_itos = remove_repeated_tokens(USPTO_src_itos,tero_rand_src_itos)
    finetune_rand_tgt_itos = remove_repeated_tokens(USPTO_tgt_itos,tero_rand_tgt_itos)
    finetune_thre64_src_itos = remove_repeated_tokens(USPTO_src_itos,tero_thre64_src_itos)
    finetune_thre64_tgt_itos = remove_repeated_tokens(USPTO_tgt_itos,tero_thre64_tgt_itos)
    finetune_thre47_src_itos = remove_repeated_tokens(USPTO_src_itos,tero_thre47_src_itos)
    finetune_thre47_tgt_itos = remove_repeated_tokens(USPTO_tgt_itos,tero_thre47_tgt_itos)

    if shared_vocab==True:
        assert finetune_rand_src_itos==finetune_rand_tgt_itos and finetune_thre64_src_itos==finetune_thre64_tgt_itos and finetune_thre47_src_itos==finetune_thre47_tgt_itos

    with open(update_vocab_path+'/random_split/vocab_share.pk', 'wb') as f:
        pickle.dump([finetune_rand_src_itos,  finetune_rand_tgt_itos], f)
    with open(update_vocab_path+'/thres_0.6_frac_0.4/vocab_share.pk', 'wb') as f:
        pickle.dump([finetune_thre64_src_itos,  finetune_thre64_tgt_itos], f)
    with open(update_vocab_path+'/thres_0.4_frac_0.7/vocab_share.pk', 'wb') as f:
        pickle.dump([finetune_thre47_src_itos,  finetune_thre47_tgt_itos], f)

    pass


def get_USTPO_finetune110_vocab(shared_vocab=True):
    '''to get USTPO finetune vocab, we need to merge the two sets:  USTPO vocab sets and terokit vocab sets

    '''
    # FIXME 这个由于字典原因，需要先产生共用的字典，然后从头训练得到USPTO的ckpts，然后（这时我们必须要求所有的finetune字典相同，才能避免用三个字典训练USPTO50K数据）
    # 只需要得到一个字典就可以
    
    
    with open('/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate/USPTO_50K/vocab_share.pk', 'rb') as f:
        USPTO_src_itos, USPTO_tgt_itos = pickle.load(f)
    with open('/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate/random_split/vocab_share_seq_10_rxnfp_euc_10.pk', 'rb') as f:
        tero_rand_src_itos, tero_rand_tgt_itos = pickle.load(f)

    finetune_rand_src_itos = remove_repeated_tokens(USPTO_src_itos,tero_rand_src_itos)
    finetune_rand_tgt_itos = remove_repeated_tokens(USPTO_tgt_itos,tero_rand_tgt_itos)

    if shared_vocab==True:
        assert finetune_rand_src_itos==finetune_rand_tgt_itos

    with open('/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate/USPTO_50K_finetune'+'/vocab_share.pk', 'wb') as f:
        pickle.dump([finetune_rand_src_itos,  finetune_rand_tgt_itos], f)

    pass

def combine_vocab(vo1_path, vo2_path, out_path, shared_vocab=True):
    with open(vo1_path, 'rb') as f:
        vo1_src_itos, vo1_tgt_itos = pickle.load(f)
    with open(vo2_path, 'rb') as f:
        vo2_src_itos, vo2_tgt_itos = pickle.load(f)

    finetune_rand_src_itos = remove_repeated_tokens(vo1_src_itos,vo2_src_itos)
    finetune_rand_tgt_itos = remove_repeated_tokens(vo1_tgt_itos,vo2_tgt_itos)

    if shared_vocab==True:
        assert finetune_rand_src_itos==finetune_rand_tgt_itos

    with open(out_path, 'wb') as f:
        pickle.dump([finetune_rand_src_itos,  finetune_rand_tgt_itos], f)

def cal_torch_model_params(model):
    '''
    :param model:
    :return:
    '''
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total_params, 'total_trainable_params': total_trainable_params}


if __name__ == "__main__":
    # with open('/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate/random_split/result/model_20000_vanilla_bs_top10_generation_untyped.pk', 'rb') as f:
    #     A = pickle.load(f)
    # print(A)
    check_vocab('/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate/thres_0.6_frac_0.4/seq_5/vocab_share.pk')

    vo1_path= '/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate/thres_0.6_frac_0.4/seq_5/vocab_share.pk'
    with open(vo1_path, 'rb') as f:
        vo1_src_itos, vo1_tgt_itos = pickle.load(f)
    add_ = ['<ENZ_6>', '<ENZ_7>', '<ENZ_8>', '<ENZ_9>', '<ENZ_10>']
    vo1_src_itos += add_
    from copy import deepcopy
    agg = deepcopy(vo1_src_itos[4:])
    vo1_src_itos = ['<unk>', '<pad>', '<sos>', '<eos>'] + sorted(agg)

    with open('/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate/thres_0.6_frac_0.4/seq_10/vocab_share.pk', 'wb') as f:
        pickle.dump([vo1_src_itos, vo1_src_itos], f)


    check_vocab('/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate/thres_0.6_frac_0.4/seq_10/vocab_share.pk')
    # check_vocab('/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate/thres_0.4_frac_0.7/esm_5/vocab_share.pk')
    # a = torch.load('/home/lishuaixin/lsx/my/retro_NP/BioNavi-NP/singlestep/vocab/terokit/thres0.6_split/.vocab.pt')
    # print(a)
    # get_USTPO_finetune110_vocab()
    # get_USTPO_finetune_vocab()
    # check_vocab('/home/lishuaixin/lsx/my/retro_NP/baseline/Retroformer/retroformer/intermediate/finetuned/thres_0.4_frac_0.7/vocab_share.pk')
    # combine_vocab('/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate/bionavi/vocab_share.pk',
    #               '/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate/USPTO_50K_finetune/vocab_share.pk',
    #               '/home/lishuaixin/lsx/my/retro_NP/Retroformer_multi_modal/retroformer/intermediate/USPTO_bionavi_finetune/vocab_share.pk')
    pass