import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_general_cf import PairwiseTrnData, PairwiseWEpochFlagTrnData, AllRankTstData
import torch as t
import torch.utils.data as data
import sys


class DataHandlerGeneralCF:
    def __init__(self):
        if configs['data']['name'] == 'amazon':
            predir = './data/amazon/'
        elif configs['data']['name'] == 'yelp':
            predir = './data/yelp/'
        elif configs['data']['name'] == 'steam':
            predir = './data/steam/'
        else:
            raise NotImplementedError
        
        self.trn_file = predir + 'trn_mat.pkl'
        self.val_file = predir + 'val_mat.pkl'
        self.tst_file = predir + 'tst_mat.pkl'


    def _load_one_mat(self, file):
        """Load one single adjacent matrix from file

        Args:
            file (string): path of the file to load

        Returns:
            scipy.sparse.coo_matrix: the loaded adjacent matrix
        """
        with open(file, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)
        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)

        return mat
    

    def _normalize_adj(self, mat):
        """Laplacian normalization for mat in coo_matrix

        Args:
            mat (scipy.sparse.coo_matrix): the un-normalized adjacent matrix

        Returns:
            scipy.sparse.coo_matrix: normalized adjacent matrix
        """
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
    

    def _make_torch_adj(self, mat, self_loop=False):
        """Transform uni-directional adjacent matrix in coo_matrix into bi-directional adjacent matrix in torch.sparse.FloatTensor
        将 coo_matrix 中的单向相邻矩阵转换为 torch.sparse.FloatTensor 中的双向相邻矩阵
        Args:
            mat (coo_matrix): the uni-directional adjacent matrix
        Returns:
            torch.sparse.FloatTensor: the bi-directional matrix
        """
        if not self_loop:
            a = csr_matrix((configs['data']['user_num'], configs['data']['user_num']))
            b = csr_matrix((configs['data']['item_num'], configs['data']['item_num']))
        else:
            data = np.ones(configs['data']['user_num'])
            row_indices = np.arange(configs['data']['user_num'])
            column_indices = np.arange(configs['data']['user_num'])
            a = csr_matrix((data, (row_indices, column_indices)), shape=(configs['data']['user_num'], configs['data']['user_num']))

            data = np.ones(configs['data']['item_num'])
            row_indices = np.arange(configs['data']['item_num'])
            column_indices = np.arange(configs['data']['item_num'])
            b = csr_matrix((data, (row_indices, column_indices)), shape=(configs['data']['item_num'], configs['data']['item_num']))

        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = self._normalize_adj(mat)

        # make torch tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])
    

    """给交互稀疏矩阵添加噪声"""
    def _random_noise_adj(self, mat):
        np.random.seed(2023)
        # 计算要修改的元素数量（即稀疏矩阵非零元素数量的50%）
        num_elements_to_modify = int(configs['noise'] * mat.nnz)
        # 随机选择要修改的位置
        modify_indices = np.random.choice(mat.nnz, num_elements_to_modify, replace=False)

        # 将选定的元素值进行翻转（将0改为1，将1改为0）
        for index in modify_indices:
            row_idx, col_idx = mat.row[index], mat.col[index]
            if mat.data[index] == 1:
                mat.data[index] = 0
            else:
                mat.data[index] = 1
        return mat

    def _sparse_matrix(self, mat):
        np.random.seed(2023)
        # 计算非零元素的总数
        total_nonzero_count = np.sum(mat != 0)
        # 计算行均值并向上取整
        row_average = np.ceil(total_nonzero_count / mat.shape[0])
        print('row_average', row_average) # 11

        # 遍历每行元素，对非零元素数量大于行均值的行，随机删除(num-avg)个非零元素，使得该行的非0元素个数等于avg。
        for i in range(mat.shape[0]):
            row_indices = mat.row == i  # 获取第i行的索引
            row_nonzero_count = np.sum(row_indices)  # 计算第i行非零元素的数量
            if row_nonzero_count > row_average * 2:
                # 获取第i行非零元素的位置索引
                nonzero_indices = np.where(row_indices)[0]
                # 随机选择要删除的非零元素索引
                delete_indices = np.random.choice(nonzero_indices, row_nonzero_count - int(row_average), replace=False)
                # 将选定的非零元素值修改为0
                mat.data[delete_indices] = 0
        return mat
    

    def load_data(self):
        ## 读取交互的稀疏矩阵  <class 'scipy.sparse._coo.coo_matrix'>
        trn_mat = self._load_one_mat(self.trn_file)

        # trn_mat = self._random_noise_adj(trn_mat)   ## 设置添加随机噪声
        # trn_mat = self._sparse_matrix(trn_mat)   ## 设置稀疏交互

        val_mat = self._load_one_mat(self.val_file)
        tst_mat = self._load_one_mat(self.tst_file)

        self.trn_mat = trn_mat
        configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape  # 获取user num 和 item num

        self.torch_adj = self._make_torch_adj(trn_mat)  # 做成torch 交互矩阵的形式
        
        if configs['model']['name'] == 'gccf':   ## 判断一下模型 
            self.torch_adj = self._make_torch_adj(trn_mat, self_loop=True)

        if configs['train']['loss'] == 'pairwise':  # 判断损失函数的类型
            trn_data = PairwiseTrnData(trn_mat)
        elif configs['train']['loss'] == 'pairwise_with_epoch_flag':
            trn_data = PairwiseWEpochFlagTrnData(trn_mat)  

        val_data = AllRankTstData(val_mat, trn_mat)  ## 所有排序的测试数据
        tst_data = AllRankTstData(tst_mat, trn_mat)

        self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.valid_dataloader = data.DataLoader(val_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)