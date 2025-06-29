import os
import pickle
import lmdb
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
import torchvision.transforms as transforms
from PIL import Image

from rdkit import Chem
# from scipy.optimize import curve_fit

import torch
from torch.utils.data import Dataset
from utils.smiles_utils import smi_tokenizer, clear_map_number, SmilesGraph
from utils.smiles_utils import canonical_smiles, canonical_smiles_with_am, remove_am_without_canonical, \
    extract_relative_mapping, get_nonreactive_mask, randomize_smiles_with_am


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean) ** 2 / (2 * standard_deviation ** 2))


class RetroDataset(Dataset):
    def __init__(self, mode, data_folder='./data', intermediate_folder='./intermediate',
                 known_class=False, shared_vocab=False, augment=False, sample=False, 
                 use_multi_modal=False,image_aug=True, args=None):
        self.data_folder = data_folder
        self.image_aug = image_aug
        self.image_default_size = args.image_default_size
        self.attri_bondlen = args.attri_bondlen
        self.model_origin = args.model_origin
        if args!=None:
            self.bond_atti_dim = args.bond_atti_dim
            # if args.USPTO_finetune:
            #     assert 'finetune' in args.intermediate_dir 
            # self.USPTO_finetune = args.USPTO_finetune

        self.enz_token = args.enz_token
        self.kmeans_label = args.kmeans_label
        self.kmeans_num_clusters = args.kmeans_num_clusters

        self.rxnfp_token = args.rxnfp_token
        self.rxnfp_class = args.rxnfp_class
        self.rxnfp_label = args.rxnfp_label
        self.rxnfp_dist = args.rxnfp_dist
        self.rxnfp_num_clusters = args.rxnfp_num_clusters
        self.rxnfp_pro = args.rxnfp_pro

        self.use_multi_modal = use_multi_modal
        assert mode in ['train', 'test', 'val']
        self.BondTypes = ['NONE', 'AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE']
        self.bondtoi = {bond: i for i, bond in enumerate(self.BondTypes)}

        self.mode = mode
        self.augment = augment
        self.known_class = known_class
        self.shared_vocab = shared_vocab
        print('Building {} data from: {}'.format(mode, data_folder))
        vocab_file = ''

        if shared_vocab:
            vocab_file += 'vocab_share.pk'
        else:
            vocab_file += 'vocab.pk'

        if mode != 'train':
            assert vocab_file in os.listdir(intermediate_folder)
            with open(os.path.join(intermediate_folder, vocab_file), 'rb') as f:
                self.src_itos, self.tgt_itos = pickle.load(f)
            self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
            self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}
            csv_path = f'{data_folder}/{mode}_enz_rxnfp'
            csv_path += '.csv'
            self.data = pd.read_csv(csv_path)
            if sample:
                self.data = self.data.sample(n=50, random_state=0)
                self.data.reset_index(inplace=True, drop=True)
        else:
            trn_csv_path = f'{data_folder}/train_enz_rxnfp'
            val_csv_path = f'{data_folder}/val_enz_rxnfp'
            trn_csv_path += '.csv'
            val_csv_path += '.csv'
            train_data = pd.read_csv(trn_csv_path)
            val_data = pd.read_csv(val_csv_path)

            if sample:# sample从而得到toy数据集方便debug
                train_data = train_data.sample(n=1000, random_state=0)
                train_data.reset_index(inplace=True, drop=True)
                val_data = val_data.sample(n=200, random_state=0)
                val_data.reset_index(inplace=True, drop=True)
            if vocab_file not in os.listdir(intermediate_folder):
                print('Building vocab...')
                raw_data = pd.concat([val_data, train_data])
                raw_data.reset_index(inplace=True, drop=True)
                prods, reacts = self.build_vocab_from_raw_data(raw_data)
                # shared_vocab src和tgt共用一个词典
                if self.shared_vocab:  # Shared src and tgt vocab
                    itos = set()
                    for i in range(len(prods)):
                        itos.update(smi_tokenizer(prods[i]))
                        itos.update(smi_tokenizer(reacts[i]))
                    itos.update(['<RX_{}>'.format(i) for i in range(1, 11)])
                    
                    itos.update(['<ENZ_unk>'])
                    itos.update(['<rxnfp_unk>'])
                    if self.enz_token:
                        itos.update(['<ENZ_{}>'.format(i) for i in range(1, self.kmeans_num_clusters+1)])
                    if self.rxnfp_token:
                        itos.update(['<rxnfp_{}>'.format(i) for i in range(1, self.rxnfp_num_clusters+1)])

                    itos.add('<UNK>')
                    itos = ['<unk>', '<pad>', '<sos>', '<eos>'] + sorted(list(itos))
                    self.src_itos, self.tgt_itos = itos, itos
                else:  # Non-shared src and tgt vocab
                    self.src_itos, self.tgt_itos = set(), set()
                    for i in range(len(prods)):
                        self.src_itos.update(smi_tokenizer(prods[i]))
                        self.tgt_itos.update(smi_tokenizer(reacts[i]))
                    self.src_itos.update(['<RX_{}>'.format(i) for i in range(1, 11)])
                    if self.enz_token:
                        itos.update(['<ENZ_{}>'.format(i) for i in range(1, self.kmeans_num_clusters+1)])
                    if self.rxnfp_token:
                        itos.update(['<rxnfp_{}>'.format(i) for i in range(1, self.rxnfp_num_clusters+1)])
                    
                    itos.update(['<ENZ_unk>'])
                    itos.update(['<rxnfp_unk>'])
                    self.src_itos.add('<UNK>')
                    self.src_itos = ['<unk>', '<pad>'] + sorted(
                        list(self.src_itos))
                    self.tgt_itos = ['<unk>', '<pad>', '<sos>', '<eos>'] + sorted(
                        list(self.tgt_itos))
                self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
                self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

                with open(os.path.join(intermediate_folder, vocab_file), 'wb') as f:
                    pickle.dump([self.src_itos, self.tgt_itos], f)
            else:
                with open(os.path.join(intermediate_folder, vocab_file), 'rb') as f:
                    self.src_itos, self.tgt_itos = pickle.load(f)

                self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
                self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

            self.data = eval('{}_data'.format(mode))

        # Build and load processed data into lmdb

        lmdb_path = f'{self.data_folder}/cooked'
        if self.enz_token:
            lmdb_path += '_{}{}'.format(self.kmeans_label,self.kmeans_num_clusters)
        if self.rxnfp_token:
            lmdb_path += '_{}_{}{}'.format(self.rxnfp_label,self.rxnfp_dist,self.rxnfp_num_clusters)
        if self.rxnfp_class:
            lmdb_path += f'_rxnfp{self.rxnfp_num_clusters}'
        lmdb_path += '_{}.lmdb'.format(self.mode)
        
        if os.path.basename(lmdb_path) not in os.listdir(self.data_folder):
            print(lmdb_path,'not in ', self.data_folder)
            self.build_processed_data(self.data)
        else: 
            print(lmdb_path,'in', self.data_folder)
        
        self.env = lmdb.open(lmdb_path,
                        max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.product_keys = list(txn.cursor().iternext(values=False))
        if sample:
            self.product_keys = self.product_keys[:200]

        self.factor_func = lambda x: (1 + gaussian(x, 5.55391565, 0.27170542, 1.20071279)) # pre-computed

    def get_itos(self,tgt=True):
        if tgt:
            return self.tgt_itos
        else:
            return self.src_itos

    def build_vocab_from_raw_data(self, raw_data): # 词典
        reactions = raw_data['reactants>reagents>production'].to_list()
        prods, reacts = [], []
        for i in tqdm(range(len(reactions))):
            rxn = reactions[i]
            r, p = rxn.split('>>')
            if not r or not p:
                continue

            src, tgt = self.parse_smi(p, r, '<UNK>', build_vocab=True) # 注意这里的输出，src为product；tgt为substrate
            if Chem.MolFromSmiles(src) is None or Chem.MolFromSmiles(tgt) is None:
                continue
            prods.append(src)
            reacts.append(tgt)
        return prods, reacts

    def build_processed_data(self, raw_data, prefix=None):
        raw_data.reset_index(inplace=True, drop=True)
        reactions = raw_data['reactants>reagents>production'].to_list()

        if self.bond_atti_dim==7:
            lmdb_path = f'{self.data_folder}/cooked'


            if self.enz_token:
                lmdb_path += '_{}{}'.format(self.kmeans_label,self.kmeans_num_clusters)
            if self.rxnfp_token:
                lmdb_path += '_{}_{}{}'.format(self.rxnfp_label,self.rxnfp_dist,self.rxnfp_num_clusters)
            if self.rxnfp_class:
                lmdb_path += f'_rxnfp{self.rxnfp_num_clusters}'
            lmdb_path += '_{}.lmdb'.format(self.mode)

            env = lmdb.open(lmdb_path,
                        map_size=1099511627776)

        else:
            assert prefix!=None
            env = lmdb.open(os.path.join(self.data_folder, '{}_{}.lmdb'.format(prefix, self.mode)),
                        map_size=1099511627776)


        with env.begin(write=True) as txn:
            print("Building {} lmdb data from csv data in: {}".format(self.mode, self.data_folder))
            for i in tqdm(range(len(reactions))):
                rxn = reactions[i]

                r, p = rxn.split('>>')
                rt = '<RX_{}>'.format(raw_data['class'][i]) if 'class' in raw_data else '<UNK>'

                if self.enz_token:
                    enz_c = '<ENZ_{}>'.format(int(raw_data[f'ENZ_{self.kmeans_label}_{self.kmeans_num_clusters}'][i])) if f'ENZ_{self.kmeans_label}_{self.kmeans_num_clusters}' in raw_data else '<ENZ_unk>'
                else:
                    enz_c = '<ENZ_unk>'
                if enz_c == '<ENZ_0>':
                    enz_c = '<ENZ_unk>'

                
                if self.rxnfp_token:
                    col_name = f'rxnfp_{self.rxnfp_num_clusters}_{self.rxnfp_dist}'
                    if self.rxnfp_pro:
                        col_name += '_pro'
                    rxnfp_c = '<rxnfp_{}>'.format(int(raw_data[col_name][i]))
                else:
                    rxnfp_c = '<rxnfp_unk>'

                if self.rxnfp_class:
                    col_name = f'rxnfp_{self.rxnfp_num_clusters}_{self.rxnfp_dist}'
                    rxnfp_c_int = int(raw_data[col_name][i])

                result = self.parse_smi_wrapper((p, r, rt, enz_c, rxnfp_c))
                if result is not None:
                    src, src_graph, tgt, context_align, nonreact_mask = result
                    graph_contents = src_graph.adjacency_matrix, src_graph.bond_type_dict, src_graph.bond_attributes

                    p_key = '{} {}'.format(i, clear_map_number(p))
                    processed = {
                        'src': src,
                        'graph_contents': graph_contents,
                        'tgt': tgt,
                        'context_align': context_align,
                        'nonreact_mask': nonreact_mask,
                        'raw_product': p,
                        'raw_reactants': r,
                        'reaction_class': rt,
                        'ENZ_class': enz_c,
                        'rxnfp_class': rxnfp_c,
                        'rxnfp_class_int': rxnfp_c_int,
                    }
                    try:
                        txn.put(p_key.encode(), pickle.dumps(processed))
                    except:
                        print('Error processing index {} and product {}'.format(i, p_key))
                        continue
                else:
                    print('Warning. Process Failed.')

        return

    def parse_smi_wrapper(self, task):
        enz_class=None
        if len(task)==5:
            prod, reacts, react_class, enz_class, rxnfp_class = task
        else:
            prod, reacts, react_class = task
        if not prod or not reacts:
            return None
        return self.parse_smi(prod, reacts, react_class, enz_class=enz_class, rxnfp_class=rxnfp_class, build_vocab=False, randomize=False)

    def parse_smi(self, prod, reacts, react_class, enz_class=None, rxnfp_class=None, build_vocab=False, randomize=False):
        ''' Process raw atom-mapped product and reactants into model-ready inputs
        :param prod: atom-mapped product
        :param reacts: atom-mapped reactants
        :param react_class: reaction class
        :param build_vocab: whether it is the stage of vocab building
        :param randomize: whether do random permutation of the reaction smiles
        :return:
        '''
        # Process raw prod and reacts:
        # print(prod)
        cano_prod_am = canonical_smiles_with_am(prod)
        cano_reacts_am = canonical_smiles_with_am(reacts)

        cano_prod = clear_map_number(prod)
        cano_reacts = remove_am_without_canonical(cano_reacts_am)

        if build_vocab:
            return cano_prod, cano_reacts

        if Chem.MolFromSmiles(cano_reacts) is None:
            cano_reacts = clear_map_number(reacts)

        if Chem.MolFromSmiles(cano_prod) is None or Chem.MolFromSmiles(cano_reacts) is None:
            return None

        if randomize:
            # print('permute product')
            cano_prod_am = randomize_smiles_with_am(prod)
            cano_prod = remove_am_without_canonical(cano_prod_am)
            if np.random.rand() > 0.5:
                # print('permute reacts')
                cano_reacts_am = '.'.join(cano_reacts_am.split('.')[::-1])
                cano_reacts = remove_am_without_canonical(cano_reacts_am)

        # Get the smiles graph
        smiles_graph = SmilesGraph(cano_prod, bond_atti_dim=self.bond_atti_dim, attri_bondlen=self.attri_bondlen)
        # Get the nonreactive masking based on atom-mapping
        gt_nonreactive_mask = get_nonreactive_mask(cano_prod_am, prod, reacts, radius=1)
        # Get the context alignment based on atom-mapping
        position_mapping_list = extract_relative_mapping(cano_prod_am, cano_reacts_am)

        # Note: gt_context_attn.size(0) = tgt.size(0) - 1; attention for token that need to predict
        gt_context_attn = torch.zeros(
            (len(smi_tokenizer(cano_reacts_am)) + 1, len(smi_tokenizer(cano_prod_am)) + 1)).long()
        # 从提取的对应位置的相对映射得到交互注意力
        for i, j in position_mapping_list:
            gt_context_attn[i][j + 1] = 1

        # Prepare model inputs
        src_token = smi_tokenizer(cano_prod)
        tgt_token = ['<sos>'] + smi_tokenizer(cano_reacts) + ['<eos>']

        src_token = [enz_class]+ [rxnfp_class] + src_token

        if self.known_class:
            src_token = [react_class] + src_token
        else:
            src_token = ['<UNK>'] + src_token
        

        gt_nonreactive_mask = [True] + gt_nonreactive_mask # 加上最前面的mask

        src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
        tgt_token = [self.tgt_stoi.get(tt, self.tgt_stoi['<unk>']) for tt in tgt_token]

        return src_token, smiles_graph, tgt_token, gt_context_attn, gt_nonreactive_mask

    def reconstruct_smi(self, tokens, src=True, raw=False):
        if src:
            if raw:
                return [self.src_itos[t] for t in tokens]
            else:
                return [self.src_itos[t] for t in tokens if t != 1]
        else:
            if raw:
                return [self.tgt_itos[t] for t in tokens]
            else:
                return [self.tgt_itos[t] for t in tokens if t not in [1, 2, 3]]

    def __len__(self):
        return len(self.product_keys)

    def image_transform(self,image): # 采用imagemol中的数据增强(finetune.py\pretrain.py 下的数据增强基本相同)

        if self.mode == 'train':
            img_transformer = [transforms.CenterCrop(self.image_default_size), transforms.RandomHorizontalFlip(),
                                    transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=360),
                                    transforms.ToTensor()]
        else:
            img_transformer = [transforms.CenterCrop(self.image_default_size), transforms.ToTensor()]
        img_transformer_use = transforms.Compose(img_transformer)
        image = img_transformer_use(image)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        image = normalize(image)

        return image
    def __getitem__(self, idx):
        p_key = self.product_keys[idx]
        with self.env.begin(write=False) as txn:
            processed = pickle.loads(txn.get(p_key))
        p_key = p_key.decode().split(' ')[1]

        p = np.random.rand()
        if self.mode == 'train' and p > 0.5 and self.augment:
            prod = processed['raw_product']
            react = processed['raw_reactants']
            rt = processed['reaction_class']
            enz_c = processed['ENZ_class']
            rxnfp_c = processed['rxnfp_class']
            rxnfp_c_int = processed['rxnfp_class_int']
            # print('prod',prod)
            # print('react',react)
            try:
                src, src_graph, tgt, context_alignment, nonreact_mask = \
                    self.parse_smi(prod, react, rt, enz_class=enz_c, rxnfp_class=rxnfp_c, randomize=True) # 这个的randomize=True，采取了随机扰动
            except:
                src, graph_contents, tgt, context_alignment, nonreact_mask = \
                    processed['src'], processed['graph_contents'], processed['tgt'], \
                    processed['context_align'], processed['nonreact_mask'] # 不使用随机扰动
                src_graph = SmilesGraph(p_key, existing=graph_contents, bond_atti_dim=self.bond_atti_dim, attri_bondlen=self.attri_bondlen)
        else:
            src, graph_contents, tgt, context_alignment, nonreact_mask = \
                processed['src'], processed['graph_contents'], processed['tgt'], \
                processed['context_align'], processed['nonreact_mask']
            src_graph = SmilesGraph(p_key, existing=graph_contents, bond_atti_dim=self.bond_atti_dim, attri_bondlen=self.attri_bondlen)
            rxnfp_c_int = processed['rxnfp_class_int']

        # Make sure the reaction class is known/unknown
        if self.known_class:
            src[0] = self.src_stoi[processed['reaction_class']]
        else:
            src[0] = self.src_stoi['<UNK>']
        if not self.model_origin:
            try:
                src[1] = self.src_stoi[processed['ENZ_class']]
            except:
                pass

            try:
                src[2] = self.src_stoi[processed['rxnfp_class']]
            except:
                pass
        
        return src, src_graph, tgt, context_alignment, nonreact_mask, rxnfp_c_int - 1


class Multi_Step_RetroDataset(Dataset):
    def __init__(self, mode,
                 known_class=False, shared_vocab=False, augment=False, 
                 args=None):

        self.model_origin = args.model_origin
        if args!=None:
            self.bond_atti_dim = args.bond_atti_dim

        self.enz_token = args.enz_token
        self.kmeans_label = args.kmeans_label
        self.kmeans_num_clusters = args.kmeans_num_clusters

        self.rxnfp_token = args.rxnfp_token
        self.rxnfp_label = args.rxnfp_label
        self.rxnfp_dist = args.rxnfp_dist
        self.rxnfp_num_clusters = args.rxnfp_num_clusters

        assert mode in ['train', 'test', 'val']
        self.BondTypes = ['NONE', 'AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE']
        self.bondtoi = {bond: i for i, bond in enumerate(self.BondTypes)}

        self.mode = mode
        self.augment = augment
        self.known_class = known_class
        self.shared_vocab = shared_vocab
        self.multi_step_smi=args.multi_step_smi
        vocab_file = '/path/to/vocab_share.pk'
        with open(os.path.join(vocab_file), 'rb') as f:
            self.src_itos, self.tgt_itos = pickle.load(f)

        self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
        self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}
    def get_itos(self,tgt=True):
        if tgt:
            return self.tgt_itos
        else:
            return self.src_itos


    def build_processed_data(self):
        r='C'
        p=self.multi_step_smi
        rt = '<UNK>'
        enz_c = '<ENZ_unk>'
        rxnfp_c = '<rxnfp_unk>'

        result = self.parse_smi_wrapper((p, r, rt, enz_c, rxnfp_c))
        if result is not None:
            src, src_graph, tgt, context_align, nonreact_mask = result
            graph_contents = src_graph.adjacency_matrix, src_graph.bond_type_dict, src_graph.bond_attributes
            self.processed = {
                'src': src,
                'p_key':clear_map_number(p),
                'graph_contents': graph_contents,
                'tgt': tgt,
                'context_align': context_align,
                'nonreact_mask': nonreact_mask,
                'raw_product': p,
                'raw_reactants': r,
                'reaction_class': rt,
                'ENZ_class': enz_c,
                'rxnfp_class': rxnfp_c,
            }

        return

    def parse_smi_wrapper(self, task):
        enz_class=None
        if len(task)==5:
            prod, reacts, react_class, enz_class, rxnfp_class = task
        else:
            prod, reacts, react_class = task
        if not prod or not reacts:
            return None
        return self.parse_smi(prod, reacts, react_class, enz_class=enz_class, rxnfp_class=rxnfp_class, build_vocab=False, randomize=False)

    def parse_smi(self, prod, reacts, react_class, enz_class=None, rxnfp_class=None, build_vocab=False, randomize=False):
        ''' Process raw atom-mapped product and reactants into model-ready inputs
        :param prod: atom-mapped product
        :param reacts: atom-mapped reactants
        :param react_class: reaction class
        :param build_vocab: whether it is the stage of vocab building
        :param randomize: whether do random permutation of the reaction smiles
        :return:
        '''
        # Process raw prod and reacts:
        cano_prod_am = canonical_smiles_with_am(prod)
        cano_reacts_am = canonical_smiles_with_am(reacts)

        cano_prod = clear_map_number(prod)
        cano_reacts = remove_am_without_canonical(cano_reacts_am)

        if build_vocab:
            return cano_prod, cano_reacts

        if Chem.MolFromSmiles(cano_reacts) is None:
            cano_reacts = clear_map_number(reacts)

        if Chem.MolFromSmiles(cano_prod) is None or Chem.MolFromSmiles(cano_reacts) is None:
            return None

        if randomize:
            # print('permute product')
            cano_prod_am = randomize_smiles_with_am(prod)
            cano_prod = remove_am_without_canonical(cano_prod_am)
            if np.random.rand() > 0.5:
                # print('permute reacts')
                cano_reacts_am = '.'.join(cano_reacts_am.split('.')[::-1])
                cano_reacts = remove_am_without_canonical(cano_reacts_am)

        # Get the smiles graph
        smiles_graph = SmilesGraph(cano_prod, bond_atti_dim=self.bond_atti_dim)
        # Get the nonreactive masking based on atom-mapping
        gt_nonreactive_mask = get_nonreactive_mask(cano_prod_am, prod, reacts, radius=1)
        # Get the context alignment based on atom-mapping
        position_mapping_list = extract_relative_mapping(cano_prod_am, cano_reacts_am)

        # Note: gt_context_attn.size(0) = tgt.size(0) - 1; attention for token that need to predict
        gt_context_attn = torch.zeros(
            (len(smi_tokenizer(cano_reacts_am)) + 1, len(smi_tokenizer(cano_prod_am)) + 1)).long()
        # 从提取的对应位置的相对映射得到交互注意力
        for i, j in position_mapping_list:
            gt_context_attn[i][j + 1] = 1

        # Prepare model inputs
        src_token = smi_tokenizer(cano_prod)
        tgt_token = ['<sos>'] + smi_tokenizer(cano_reacts) + ['<eos>']

        src_token = [enz_class]+ [rxnfp_class] + src_token

        if self.known_class:
            src_token = [react_class] + src_token
        else:
            src_token = ['<UNK>'] + src_token
        
        gt_nonreactive_mask = [True] + gt_nonreactive_mask # 加上最前面的mask

        src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]
        tgt_token = [self.tgt_stoi.get(tt, self.tgt_stoi['<unk>']) for tt in tgt_token]

        return src_token, smiles_graph, tgt_token, gt_context_attn, gt_nonreactive_mask

    def reconstruct_smi(self, tokens, src=True, raw=False):
        if src:
            if raw:
                return [self.src_itos[t] for t in tokens]
            else:
                return [self.src_itos[t] for t in tokens if t != 1]
        else:
            if raw:
                return [self.tgt_itos[t] for t in tokens]
            else:
                return [self.tgt_itos[t] for t in tokens if t not in [1, 2, 3]]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        self.build_processed_data()

        src, graph_contents, tgt, context_alignment, nonreact_mask = \
                self.processed['src'], self.processed['graph_contents'], self.processed['tgt'], \
                self.processed['context_align'], self.processed['nonreact_mask']
        src_graph = SmilesGraph(self.processed['p_key'], existing=graph_contents, bond_atti_dim=self.bond_atti_dim)

        # Make sure the reaction class is known/unknown
        if self.known_class:
            src[0] = self.src_stoi[self.processed['reaction_class']]
        else:
            src[0] = self.src_stoi['<UNK>']
        if not self.model_origin:
            try:
                src[1] = self.src_stoi[self.processed['ENZ_class']]
            except:
                pass

            try:
                src[2] = self.src_stoi[self.processed['rxnfp_class']]
            except:
                pass

        return src, src_graph, tgt, context_alignment, nonreact_mask