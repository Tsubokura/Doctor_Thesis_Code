

import re
import random
import argparse
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EchoStateNetwork(nn.Module):
    def __init__(self, model_params, dataset_params):
        super(EchoStateNetwork, self).__init__()
        
        self.reservoir_size = int(model_params["reservoir_size"])
        self.reservoir_weights_scale = float(model_params["reservoir_weights_scale"])
        
        self.input_size = int(model_params["input_size"])
        self.channel_size = int(model_params["channel_size"])
        self.input_weights_scale = float(model_params["input_weights_scale"])
        self.spectral_radius = float(model_params["spectral_radius"])
        self.density = float(model_params["reservoir_density"])
        self.reak_rate = float(model_params["reak_rate"])
        
        self.sequence_length = int(int(dataset_params["sequence_length"]) / int(dataset_params["slicing_size"]))

        # リザバー結合行列 (ランダムに初期化)
        self.register_parameter("reservoir_weights", torch.nn.Parameter(torch.empty((self.reservoir_size, self.reservoir_size)).uniform_(-self.reservoir_weights_scale, self.reservoir_weights_scale).to(device), requires_grad=False))
        
        # リザバー入力行列 (ランダムに初期化)
        self.register_parameter("input_weights", torch.nn.Parameter(torch.empty((self.reservoir_size, self.input_size * self.channel_size)).uniform_(-self.input_weights_scale, self.input_weights_scale).to(device), requires_grad=False))

        #リザバー結合のスパース処理
        self.reservoir_weights_mask = torch.empty((self.reservoir_size, self.reservoir_size)).uniform_(0, 1)
        self.reservoir_weights_mask = torch.where(self.reservoir_weights_mask < self.density, torch.tensor(1), torch.tensor(0)).to(device)
        self.reservoir_weights *= self.reservoir_weights_mask
        
        #スペクトル半径の処理
        _, singular_values, _ = torch.svd(self.reservoir_weights)
        rho_reservoir_weights = torch.max(singular_values).item()
        self.reservoir_weights *= self.spectral_radius / rho_reservoir_weights
        
       #最終時刻における，リザバー状態ベクトル
        self.last_reservoir_state_matrix = torch.zeros(self.channel_size, self.reservoir_size).to(device)
    

    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(2)
        
        # 各シーケンスのバッチに対してリザバー状態を初期化
        self.reservoir_state_matrix = torch.zeros(batch_size, self.channel_size,  sequence_length, self.reservoir_size).to(device)

        self.last_reservoir_state_matrix = torch.zeros(
                batch_size, self.channel_size, self.reservoir_size, device=device
            )
        
        for t in range(sequence_length):
            input_at_t = torch.matmul(x[:, :, t, :].reshape(batch_size, -1), self.input_weights.t())
            input_at_t = input_at_t.unsqueeze(1)
            if t == 0:
                state_update = torch.matmul(self.last_reservoir_state_matrix, self.reservoir_weights)
            else:
                state_update = torch.matmul(self.reservoir_state_matrix[:, :, t-1, :], self.reservoir_weights)
            self.reservoir_state_matrix[:, :, t, :] = self.reak_rate * torch.tanh(input_at_t + state_update) + \
                                                    (1 - self.reak_rate) * self.reservoir_state_matrix[:, :, t-1, :]

        self.last_reservoir_state_matrix = self.reservoir_state_matrix[:, :, -1, :]
        return self.reservoir_state_matrix

    def reset_hidden_state(self):
        self.last_reservoir_state_matrix = torch.zeros(self.channel_size, self.reservoir_size, device=device)
        # pass
        # print("内部状態がリセットされました")
    
#リードアウト層
class ReadOut(nn.Module):
    def __init__(self, model_params, dataset_params):
        super(ReadOut, self).__init__()
        # self.reservoir_state_matrix_size = int(model_params["reservoir_size"]) + int(model_params["input_size"]) + 1
        self.reservoir_state_matrix_size = int(model_params["reservoir_size"])
        self.output_size = int(model_params["ReadOut_output_size"])
        self.batch_training = model_params["Batch_Training"]
        self.channel_size = int(model_params["channel_size"])
        self.sequence_length = int(int(dataset_params["sequence_length"]) / int(dataset_params["slicing_size"]))
        
        self.readout_dense = nn.Linear(self.reservoir_state_matrix_size, self.output_size, bias=False)
        
        #線形回帰におけるバッチ学習を行うならば，リードアウト層を最急降下法による学習対象にしない
        if self.batch_training == True:
            self.readout_dense.weight.requires_grad = False
        else:
            None
            
        nn.init.xavier_uniform_(self.readout_dense.weight)
        
    def forward(self, x):
        output = self.readout_dense(x)
        return output
    
    # リッジ回帰によるリードアウトの導出
    @staticmethod
    def ridge_regression(X, Y, alpha):
        # データ行列 X の形状を取得

        X = X.squeeze()
        Y = Y.squeeze()

        p, n  = X.shape

        # 正則化項の行列を作成
        ridge_matrix = (alpha * torch.eye(n)).float().to(device)

        X = X.float()
        Y = Y.float()
        # リッジ回帰の係数を計算
        coefficients = torch.linalg.solve(torch.matmul(X.T, X) + ridge_matrix, torch.matmul(X.T, Y)).T

        return coefficients

    @staticmethod
    def ridge_regression_update(outputs, targets, model, alpha=0):
        with torch.no_grad():
            # リッジ回帰を用いて重みを求める
            # モデルのパラメータを更新
            new_weights = ReadOut.ridge_regression(outputs.squeeze(), targets.squeeze(), alpha)

            # モデルのパラメータを更新
            model.ReadOut.readout_dense.weight.copy_(new_weights)
        
        return None
    #それぞれのモデルパラメータ候補を辞書に格納する
    @staticmethod
    def model_params_candinate(model_params):
        model_params_combinations = list(itertools.product(*model_params.values()))
        param_dicts = [dict(zip(model_params.keys(), combination)) for combination in model_params_combinations]
        
        return param_dicts

    #モデル構造を辞書型に格納
    @staticmethod
    def model_sturcture_dict(model):
        layers_dict = {}
        for name, module in model.named_modules():
            layers_dict[name] = {
                'type': type(module).__name__,
                'parameters': {p: getattr(module, p) for p in module.__dict__ if not p.startswith('_')}
            }
        
        #モデル名と初期の引数は削除
        del(layers_dict[''])
        
        return layers_dict

class ESN(nn.Module):
    def __init__(self, model_params, training_params, dataset_params):
        super(ESN, self).__init__()
        self.ESN = EchoStateNetwork(model_params, dataset_params)
        self.ReadOut = ReadOut(model_params, dataset_params)
    
    def forward(self, x):
        self.Reservoir_State_Matrix = self.ESN(x)
        self.ReadOut_Reservoir = self.ReadOut(self.Reservoir_State_Matrix)

        #channle_size次元はいらないので，減らす
        self.ReadOut_Reservoir = self.ReadOut_Reservoir.squeeze(1)

        return self.ReadOut_Reservoir

    def reset_hidden_state(self):
        self.ESN.reset_hidden_state()