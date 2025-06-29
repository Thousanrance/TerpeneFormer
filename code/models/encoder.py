import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from models.module import PositionwiseFeedForward, LayerNorm, MultiHeadedAttention
from torch.nn.parameter import Parameter

class Imagemol_resnet18(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        model = torchvision.models.resnet18(pretrained=False)
        # if self.multimodal_input:
        #     self.n_features = int(model.fc.in_features/2) # 256,多模态的数据方便concatenate
        #     # print(f'self.n_features {self.n_features}')
        # else:
        self.n_features = model.fc.in_features  # 512

        imagemol_ckpt='/path/to/MolScribe/encoder_ckpt/ImageMol.pth.tar'

        if os.path.isfile(imagemol_ckpt):  # only support ResNet18 when loading resume
            # print("=> loading checkpoint '{}'".format(imagemol_ckpt))
            checkpoint = torch.load(imagemol_ckpt)
            ckp_keys = list(checkpoint['state_dict'])
            # print(len(ckp_keys))
            # print(ckp_keys)
            cur_keys = list(model.state_dict())
            model_sd = model.state_dict()
            ckp_keys = ckp_keys[:120]
            cur_keys = cur_keys[:120]
            # print(ckp_keys)
            for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                model_sd[cur_key] = checkpoint['state_dict'][ckp_key] # 将当前key和ckpt中的key对应的值进行替换

            model.load_state_dict(model_sd) # 载入相关参数
            # arch = checkpoint['arch']
            # print("resume model info: arch: {}".format(arch))
        else:
            print("=> no checkpoint found at '{}'".format())
        if not args.new_trans:
            model.avgpool = nn.Identity()
        model.fc = nn.Identity()
        self.cnn=model 
        # print(model)
    def forward(self, x):
        self.cnn = self.cnn.to(x.device)
        features = self.cnn(x) # (B,512)
        if self.args.new_trans:
            features = features.view(-1,512)
        else:
            features = features.view(-1,512,7,7)
        return features

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attn):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = attn
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask, edge_feature, pair_indices):
        input_norm = self.layer_norm(inputs)
        context, attn, edge_feature_updated = self.self_attn(input_norm, input_norm, input_norm, mask=mask,
                                                             edge_feature=edge_feature,
                                                             pair_indices=pair_indices)

        out = self.dropout(context) + inputs
        if edge_feature is not None:
            edge_feature = self.layer_norm(edge_feature + edge_feature_updated)
        return self.feed_forward(out), attn, edge_feature

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, embeddings_bond, attn_modules,args=None):
        super(TransformerEncoder, self).__init__()

        self.args = args
        self.use_multi_modal = args.use_multi_modal
        self.use_multi_modal_after = args.use_multi_modal_after
        self.after_big_imgdim = args.after_big_imgdim

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.embeddings_bond = embeddings_bond
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout, attn_modules[i])
             for i in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)
        # TODO
        
        
        if args.after_big_imgdim:
            self.avp = nn.AvgPool2d(kernel_size=7)
            self.fc2 = nn.Linear(49,1)
            self.conv1x1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1) 
        else:
            if self.args.new_trans and self.args.residual_connect: # TODO zero init params
                self.fc3 = nn.Linear(512,256, bias=False)
                if self.args.zero_init:
                    w=torch.normal(0, 0.001, size=(256,512), requires_grad=True)     #生成与权重大小相同的tensor
                    self.fc3.weight = Parameter(w) 
            else:
                self.fc1 = nn.Linear(512,1) # 这种选择是7*7*512->7*7
            
    def forward(self, src, bond=None):
        # encoder 接受序列和键矩阵输入；bond为键，因为将src_len 每个token视为原子，因此键的邻接矩阵为形状(src_len,src_len),7为键性质
        '''
        :param src: [src_len, batch_size]
        :param bond: [batch_size, src_len, src_len, 7]
        :return:
        '''
        global node_feature
        # print('src',src.shape)
        emb = self.embeddings(src)
        # print('emb',emb.shape)
        out = emb.transpose(0, 1).contiguous()
        # print('out',out.shape)
        # print('out0',out.shape)
        if bond is not None:
            # print('bond',bond.shape)
            # valid_items 相当于把原来的进行满足调节的tensor堆叠起来，只保留堆叠的维度（valid_items）和最后一个维度
            # # 最后一个维度求和，得到成键的索引(valid_items, valid_items, valid_items),
            pair_indices = torch.where(bond.sum(-1) > 0) 
            valid_bond = bond[bond.sum(-1) > 0] # 有效键，否则其它的为无效token的连接(valid_items,7)
            # print('valid_bond',valid_bond.shape)
            edge_feature = self.embeddings_bond(valid_bond.float()) # (valid_items,256)
            # print('edge_feature',edge_feature.shape)
        else:
            pair_indices, edge_feature = None, None

        words = src.transpose(0, 1)
        w_batch, w_len = words.size()
        # print('words',words.shape)
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)
        # print('mask',mask.shape)
        # Run the forward pass of every layer of the transformer.
        for i in range(self.num_layers):
            out, attn, edge_feature = self.transformer[i](out, mask, edge_feature, pair_indices)

        out = self.layer_norm(out)
        out = out.transpose(0, 1).contiguous()
        # print('out',out.shape) # out (seq_len,bs,256)
        image_features = None
        edge_out = self.layer_norm(edge_feature) if edge_feature is not None else None
        # print('edge_out',edge_out.shape)
        return out, image_features, edge_out



