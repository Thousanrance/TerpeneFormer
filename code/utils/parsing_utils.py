import argparse

def get_translate_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda', help='device GPU/CPU')
    parser.add_argument('--batch_size_val', type=int, default=4, help='batch size')
    parser.add_argument('--batch_size_trn', type=int, default=4, help='batch size')
    parser.add_argument('--beam_size', type=int, default=10, help='beam size')
    parser.add_argument('--attri_bondlen', action='store_true')

    parser.add_argument('--encoder_num_layers', type=int, default=8, help='number of layers of transformer')
    parser.add_argument('--decoder_num_layers', type=int, default=8, help='number of layers of transformer')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
    parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
    parser.add_argument('--d_ff', type=int, default=2048, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')


    # parser.add_argument('--stepwise', type=str, default='False', choices=['True', 'False'], help='')
    # parser.add_argument('--use_template', type=str, default='False', choices=['True', 'False'], help='')
    # parser.add_argument('--known_class', type=str, default='True', help='with reaction class known/unknown')
    # parser.add_argument('--shared_vocab', type=str, default='False', choices=['True', 'False'], help='whether sharing vocab')
    # parser.add_argument('--shared_encoder', type=str, default='False', choices=['True', 'False'],
    #                     help='whether sharing encoder')

    parser.add_argument('--stepwise',action='store_true', help='')
    parser.add_argument('--use_template', action='store_true', help='')
    parser.add_argument('--known_class', action='store_true', help='with reaction class known/unknown')
    parser.add_argument('--shared_vocab', action='store_true', help='whether sharing vocab')
    parser.add_argument('--shared_encoder', action='store_true',
                        help='whether sharing encoder')
    parser.add_argument('--use_multi_modal', action='store_true')
    parser.add_argument('--use_multi_modal_front', action='store_true', help='wether to add image modal in the front or not')
    parser.add_argument('--use_multi_modal_after', action='store_true', help='concate picture dim in the hidden layer, \
                        and then use a FC layer to reduce dimention ')
    parser.add_argument('--after_concate_layer', type=int, help='the layer of  concate picture dim in the hidden layer')
    parser.add_argument('--bond_atti_dim',type=int, default=7)
    parser.add_argument('--image_default_size', type=int, default=224)
    parser.add_argument('--after_big_imgdim', action='store_true',help='whther to choose big img dim 512 or not')
    parser.add_argument('--residual_connect', action='store_true')
    parser.add_argument('--cos_sim', action='store_true')

    parser.add_argument('--new_trans', action='store_true')
    parser.add_argument('--zero_init', action='store_true')
    parser.add_argument('--ckpt_range', action='store_true')
    parser.add_argument('--range_begin', type=int, default=None)
    parser.add_argument('--range_end', type=int, default=None)
    parser.add_argument('--range_step', type=int, default=2500)
    parser.add_argument('--ckpt_re_pt', action='store_true')

    parser.add_argument('--steps', type=int)
    parser.add_argument('--sample', action='store_true')

    parser.add_argument('--data_dir', type=str, default='/path/to/TeroRXN/random_split', help='base directory')
    parser.add_argument('--intermediate_dir', type=str, default='/path/to/intermediate', help='intermediate directory')
    parser.add_argument('--checkpoint_dir', type=str, default='/path/to/ckpts', help='checkpoint directory')
    parser.add_argument('--checkpoint', type=str, help='checkpoint model file')

    parser.add_argument('--USPTO_finetune', action='store_true', help='whether to use USPTO50K data to init our checkpoints params and finetune on our terokit dataset')
    parser.add_argument('--USPTO_extend_vocab', action='store_true')
    parser.add_argument('--optim_loadstate', action='store_true')

    # enzymes
    parser.add_argument('--enz_token', action='store_true')
    parser.add_argument('--kmeans_label', type=str, default='seq', choices=['seq', 'esm'])
    parser.add_argument('--kmeans_num_clusters', type=int, default=10)

    # rxnfp
    parser.add_argument('--rxnfp_class', action='store_true')
    parser.add_argument('--rxnfp_token', action='store_true')
    parser.add_argument('--rxnfp_label', type=str, default='rxnfp')
    parser.add_argument('--rxnfp_dist', type=str, default='euc', choices=['euc','cos'])
    parser.add_argument('--rxnfp_num_clusters', type=int, default=10)
    parser.add_argument('--model_origin', action='store_true')
    parser.add_argument('--rxnfp_pro', action='store_true')
    
    parser.add_argument('--get_val_result', action='store_true')
    parser.add_argument('--multi_step_pred', action='store_true')
    parser.add_argument('--multi_step_smi', type=str)
    args = parser.parse_args()

    return args