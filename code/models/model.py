from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import math
import torch
import torch.nn as nn
import numpy as np
import os


from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from models.embedding import Embedding
from models.module import MultiHeadedAttention


class RetroModel(nn.Module):
    def __init__(self, encoder_num_layers, decoder_num_layers, d_model, heads, d_ff, dropout,
                 vocab_size_src, vocab_size_tgt, shared_vocab, shared_encoder=False, src_pad_idx=1,
                 tgt_pad_idx=1, args=None):
        super(RetroModel, self).__init__()

        self.use_multi_modal=args.use_multi_modal
        self.use_multi_modal_front=args.use_multi_modal_front
        self.use_multi_modal_after=args.use_multi_modal_after

        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.shared_vocab = shared_vocab
        self.shared_encoder = shared_encoder
        self.rxn_class_num = args.rxnfp_num_clusters

        if shared_vocab:
            assert vocab_size_src == vocab_size_tgt and src_pad_idx == tgt_pad_idx
            # if USPTO_extend_vocab
            self.embedding_src = self.embedding_tgt = Embedding(vocab_size=vocab_size_src + 1, embed_size=d_model,
                                                                padding_idx=src_pad_idx, USPTO_extend_vocab=args.USPTO_extend_vocab)
        else:
            self.embedding_src = Embedding(vocab_size=vocab_size_src + 1, embed_size=d_model, padding_idx=src_pad_idx,
                                           USPTO_extend_vocab=args.USPTO_extend_vocab)
            self.embedding_tgt = Embedding(vocab_size=vocab_size_tgt + 1, embed_size=d_model, padding_idx=tgt_pad_idx,
                                           USPTO_extend_vocab=args.USPTO_extend_vocab)

        self.bond_atti_dim=args.bond_atti_dim
        
        self.embedding_bond = nn.Linear(self.bond_atti_dim, d_model)

        multihead_attn_modules_en = nn.ModuleList(
            [MultiHeadedAttention(heads, d_model, dropout=dropout)
             for _ in range(encoder_num_layers)])
        if shared_encoder:
            assert encoder_num_layers == decoder_num_layers
            multihead_attn_modules_de = multihead_attn_modules_en
        else:
            multihead_attn_modules_de = nn.ModuleList(
                [MultiHeadedAttention(heads, d_model, dropout=dropout)
                 for _ in range(decoder_num_layers)])

        self.encoder = TransformerEncoder(num_layers=encoder_num_layers,
                                          d_model=d_model, heads=heads,
                                          d_ff=d_ff, dropout=dropout,
                                          embeddings=self.embedding_src,
                                          embeddings_bond=self.embedding_bond,
                                          attn_modules=multihead_attn_modules_en,
                                          args=args)
        # 288+280=568
        # 
        self.multi_modal_trans = nn.Sequential(nn.Linear(568, 568),
                                                nn.Sigmoid(),
                                                nn.Linear(568, 280),
                                                nn.Sigmoid(),
                                                nn.Dropout(dropout))


        self.decoder = TransformerDecoder(num_layers=decoder_num_layers,
                                          d_model=d_model, heads=heads,
                                          d_ff=d_ff, dropout=dropout,
                                          embeddings=self.embedding_tgt,
                                          self_attn_modules=multihead_attn_modules_de,
                                          args=args)

        self.atom_rc_identifier = nn.Sequential(nn.Linear(d_model, 1),
                                                nn.Sigmoid())
        self.bond_rc_identifier = nn.Sequential(nn.Linear(d_model, 1),
                                                nn.Sigmoid())

        
        self.generator = nn.Sequential(nn.Linear(d_model, vocab_size_tgt),
                                    nn.LogSoftmax(dim=-1))

        self.softmax = nn.Softmax(dim=-1)

        # 分类头
        self.rxn_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, self.rxn_class_num),
        )

    def forward(self, src, tgt, bond=None, teacher_mask=None):
        # print('src',src.shape) # (src_len,bs)
        # print('tgt',tgt.shape) # (tgt_len,bs)
        encoder_out, image_features, edge_feature = self.encoder(src, bond)

        atom_rc_scores = self.atom_rc_identifier(encoder_out) # (bs,len,1)
        # print('encoder_out',encoder_out.shape) # (seq_len,bs,dim=256)
        # print('atom_rc_scores',atom_rc_scores.shape)   # (seq_len,bs,1)
        bond_rc_scores = self.bond_rc_identifier(edge_feature) if edge_feature is not None else None
        # print('edge_feature',edge_feature.shape) # (valid_items,256)

        rxn_class_scores = self.rxn_classifier(encoder_out.mean(dim=0))  # (bs, rxn_num_class)

        if teacher_mask is None:  # Naive Inference
            student_mask = self.infer_reaction_center_mask(bond, atom_rc_scores, bond_rc_scores)
            # print('student_mask',student_mask.shape)
            decoder_out, top_aligns = self.decoder(src, tgt[:-1], encoder_out, student_mask.clone())

        else:  
            decoder_out, top_aligns = self.decoder(src, tgt[:-1], encoder_out, teacher_mask.clone())
            # print('teacher_mask',teacher_mask.shape) # (src_seq_len,bs)
            # print('decoder_out',decoder_out.shape) # (tgt_seq_len,bs,dim=256)
            # print('top_aligns',len(top_aligns),'  ', top_aligns[0].shape) # num_layers, (bs, tgt_seq_len, src_seq_len)

        generative_scores = self.generator(decoder_out)
        # print('generative_scores',generative_scores.shape) # (tgt_seq_len,bs,vocab_size_tgt)
        return generative_scores, atom_rc_scores, bond_rc_scores, top_aligns, rxn_class_scores

    @staticmethod
    def infer_reaction_center_mask(bond, atom_rc_scores, bond_rc_scores=None):
        atom_rc_scores = atom_rc_scores.squeeze(2)
        if bond_rc_scores is not None:
            bond_rc_scores = bond_rc_scores.squeeze(1)
            bond_indicator = torch.zeros((bond.shape[0], bond.shape[1], bond.shape[2])).bool().to(bond.device)
            bond_indicator[bond.sum(-1) > 0] = (bond_rc_scores > 0.5)

            result = (~(bond_indicator.sum(dim=1).bool()) + ~(bond_indicator.sum(dim=2).bool()) +
                      (atom_rc_scores.transpose(0, 1) < 0.5)).transpose(0, 1)
        else:
            result = (atom_rc_scores.transpose(0, 1) < 0.5).transpose(0, 1)
        return result
