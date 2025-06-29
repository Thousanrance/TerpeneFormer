import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from models.module import PositionwiseFeedForward, LayerNorm, MultiHeadedAttention
import numpy as np
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout,
                 self_attn, context_attn, args=None):
        super(TransformerDecoderLayer, self).__init__()

        self.args = args
        self.after_big_imgdim = None
        if args !=None:
            self.after_big_imgdim = args.after_big_imgdim
            self.residual_connect = args.residual_connect
            self.cos_sim = args.cos_sim
        if self.args.new_trans and not self.args.residual_connect:
            self.new_multi_modal_after_trans = nn.Linear(768, d_model, bias=False)
            if self.args.zero_init:
                w=torch.normal(0, 0.001, size=(d_model, 768), requires_grad=True)     #生成与权重大小相同的tensor
                self.new_multi_modal_after_trans.weight = Parameter(w) 
        else:
            if self.after_big_imgdim:
                # 512+256=768
                self.use_multi_modal_after_trans = nn.Sequential(nn.Linear(768, d_model),nn.ReLU(),nn.Linear(d_model, d_model),nn.ReLU(),nn.Dropout(dropout))
            else:
                # 7*7+256=305
                self.use_multi_modal_after_trans = nn.Sequential(nn.Linear(305, d_model),nn.ReLU(),nn.Linear(d_model, d_model),nn.ReLU(),nn.Dropout(dropout))
                
        self.self_attn = self_attn
        self.context_attn = context_attn
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(5000)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        # 有时候会遇到self.register_buffer('name', Tensor)的操作，该方法的作用是定义一组参数，
        # 该组参数的特别之处在于：模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），
        # 但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
        self.register_buffer('mask', mask) 

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                nonreactive_mask_input=None, layer_input=None, layer_cache=None, 
                image_features=None, use_multi_modal_after=False):
        # 训练的时候为tgt_len（但是会使用mask） ，预测的时候取第step步则为1
        # inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
        # memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
        # src_pad_mask (`LongTensor`): `[batch_size x tgt_len x src_len]`
        # infer_decision_input (`LongTensor`): `[batch_size x tgt_len]`
        # nonreactive_mask_input (`BoolTensor`): `[batch_size x src_len]`
        # torch.gt(a,b)函数比较a中元素大于（这里是严格大于）b中对应元素，大于则为1，不大于则为0，
        # 这里a为Tensor，b可以为与a的size相同的Tensor或常数。
        # mask 为True代表就真的遮蔽了，因此padding字符需要遮蔽，然后就是每一步的自回归需要遮蔽
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                            :tgt_pad_mask.size(1)], 0) #  (bs,tgt_len,tgt_len)
        # print('inputs',inputs.shape)
        if self.cos_sim:
            inputs_origin = inputs.clone()
            input_norm_origin = self.layer_norm_1(inputs_origin)
        if use_multi_modal_after:
            # print('image_features',image_features.shape) # (bs, 49 or 512)
            if len(image_features.size())==3:
                img_f = image_features.repeat(1,inputs.size(1),1)
            else:
                img_f = image_features.unsqueeze(1).repeat(1,inputs.size(1),1)
            # print('img_f',img_f.shape) # (bs, tgt_len, 49 or 512)
            # print('inputs',inputs.shape) # (bs, tgt_len, model_dim)
            if self.residual_connect:
                if self.args.new_trans:
                    # print('new_trans, resi')
                    inputs = inputs+img_f
                else:
                    inputs_cat = torch.cat((inputs,img_f),dim=2)
                    inputs_cat_trans = self.use_multi_modal_after_trans(inputs_cat)
                    inputs = inputs+inputs_cat_trans
            else:
                if self.args.new_trans:
                    inputs_cat = torch.cat((inputs,img_f),dim=2)
                    # print('inputscat', inputs_cat.shape)
                    inputs = self.new_multi_modal_after_trans(inputs_cat)
                else:
                    inputs_cat = torch.cat((inputs,img_f),dim=2)
                    inputs = self.use_multi_modal_after_trans(inputs_cat)

        input_norm = self.layer_norm_1(inputs)
        if self.cos_sim and use_multi_modal_after:
            cos_all = nn.CosineSimilarity(dim=2, eps=1e-6)
            cos_flatten =  nn.CosineSimilarity(dim=0, eps=1e-6)
            # print('inputs all', cos_all(inputs_origin, inputs))
            print('inputs flatten', cos_flatten(inputs_origin.flatten(), inputs.flatten()))
            # print('inputs_origin',inputs_origin)
            # print('inputs',inputs)
            # if abs(cos_flatten(inputs_origin.flatten(), inputs.flatten())-1)>0.01:

            # print('inputs norm all', cos_all(input_norm, input_norm_origin))
            # print('inputs norm flatten', cos_flatten(input_norm.flatten(), input_norm_origin.flatten()))
        # Self-attention:
        all_input = input_norm
        if layer_input is not None: # 训练的时候使用了mask，模拟了自回归的过程，但是推理的时候需要将已经解码的过程输出来
            all_input = torch.cat((layer_input, input_norm), dim=1)
            dec_mask = None
        # print('all_input',all_input.shape) # [batch_size x tgt_len x model_dim]
        '''区分self还是context只有在推理过程会用到, 否则正常训练不会区分self还是context
        '''
        # 自注意力query来自于src
        query, self_attn, _ = self.self_attn(all_input, all_input, input_norm,
                                             mask=dec_mask,
                                             type="self",
                                             layer_cache=layer_cache)
        # print('query',query.shape) # [batch_size x tgt_len x model_dim]
        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)

        # Context-attention:
        mid, context_attn, _ = self.context_attn(memory_bank, memory_bank, query_norm,
                                                 mask=src_pad_mask,
                                                 additional_mask=nonreactive_mask_input,
                                                 type="context",
                                                 layer_cache=layer_cache)
        output = self.feed_forward(self.drop(mid) + query)
        # print('output',output.shape) # [batch_size x tgt_len x model_dim]
        # print('context_attn',context_attn.shape) # [batch_size x tgt_len x src_len]
        return output, context_attn, all_input

    def _get_attn_subsequent_mask(self, size):
        attn_shape = (1, size, size)
        #  k代表第k个对角线，k=0代表主对角线，k=1代表从主对角线以上（不含）为ones 
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, self_attn_modules, args=None):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.num_layers = num_layers
        self.embeddings = embeddings

        if args !=None:
            self.use_multi_modal_after = args.use_multi_modal_after
            self.after_concate_layer = args.after_concate_layer

        context_attn_modules = nn.ModuleList(
            [MultiHeadedAttention(heads, d_model, dropout=dropout)
             for _ in range(num_layers)])

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout, self_attn_modules[i], context_attn_modules[i],
                                     args=args)
             for i in range(num_layers)])

        self.layer_norm_0 = LayerNorm(d_model)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, src, tgt, memory_bank, nonreactive_mask=None, state_cache=None, step=None):
        '''
        :param src:
        :param tgt:
        :param memory_bank:
        :param nonreactive_mask: mask corresponding to reaction center identification from encoder
        :param infer_label: only occur in training for teacher's forcing; during inference, infer_label is the infer_decision.
        :param state_cache:
        :param step:
        :return:
        '''
        if nonreactive_mask is not None:
            nonreactive_mask[0] = False     # allow attention to the initial src token

        src_words = src.transpose(0, 1)
        tgt_words = tgt.transpose(0, 1)
        # print('src',src.shape) # (src_len,bs)
        # print('tgt',tgt.shape) # (tgt_len,bs)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Initialize return variables.
        outputs = []

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim
        if step is not None:
            tgt_words = tgt[-1].unsqueeze(0).transpose(0, 1)
            # print('tgt_words',tgt_words.shape)
            tgt_batch, tgt_len = tgt_words.size()

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        # assume src padding idx and tgt padding idx are the same
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len)
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)


        # (src_seq_len,bs)
        nonreactive_mask_input = nonreactive_mask.transpose(0, 1) if nonreactive_mask is not None else None
        top_context_attns = []
        for i in range(self.num_layers):
            if i == self.after_concate_layer or  self.after_concate_layer==8:
                use_multi_modal_after = True
            else:
                use_multi_modal_after = False

            layer_input = None
            layer_cache = {'self_keys': None,
                           'self_values': None,
                           'memory_keys': None,
                           'memory_values': None}
            if state_cache is not None: # 如果输入的是{}，不是None
                # print('before: state_cache is not None')
                layer_cache = state_cache.get('layer_cache_{}'.format(i), layer_cache)
                # print(layer_cache)
                layer_input = state_cache.get('layer_input_{}'.format(i), layer_input)
            # print('output',output.shape)
            output, top_context_attn, all_input \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    src_pad_mask, tgt_pad_mask,
                    layer_input=layer_input,
                    layer_cache=layer_cache,
                    nonreactive_mask_input=nonreactive_mask_input,
                    use_multi_modal_after=use_multi_modal_after)

            top_context_attns.append(top_context_attn)
            if state_cache is not None:
                # print('after: state_cache is not None')
                state_cache['layer_cache_{}'.format(i)] = layer_cache
                # print(layer_cache)
                state_cache['layer_input_{}'.format(i)] = all_input


        output = self.layer_norm(output)
        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()

        return outputs, top_context_attns
