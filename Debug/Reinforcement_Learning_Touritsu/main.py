import random
import time
import re
import multiprocessing
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

import matplotlib.pyplot as plt

from PIL import Image
from collections import namedtuple, deque

from ESN import ESN
from ESN_Agent import ESNAgent, ESNTrainer, Observer
from Replay_Memory import RecurrentExperienceReplayMemory

import wandb

WANDB_API_KEY = "2d996a98ef8dddefa91d675f85b5efd96fb911ae"  # あなたのWandB APIキーをここに入力してください

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def model_params_candinate(model_params):
    model_params_combinations = list(itertools.product(*model_params.values()))
    param_dicts = [dict(zip(model_params.keys(), combination)) for combination in model_params_combinations]
    return param_dicts

# モデル構造を辞書型に格納
def model_sturcture_dict(model):
    layers_dict = {}
    for name, module in model.named_modules():
        layers_dict[name] = {
            'type': type(module).__name__,
            'parameters': {p: getattr(module, p) for p in module.__dict__ if not p.startswith('_')}
        }
    # モデル名と初期の引数は削除
    del(layers_dict[''])
    return layers_dict

class CartPoleObserver(Observer):
    def transform(self, state):
        return np.array(state)

def main(play, is_test):
    trial_number = 1
    set_seed(42)

    model_params = {
        "reservoir_size" : [100],
        "channel_size" : [1],
        "reservoir_weights_scale" : [1],
        "input_weights_scale" : [10],
        "spectral_radius" : [0.95],
        "reservoir_density" : [0.2],
        "reak_rate" : [0.5],
        "Batch_Training" : [False]
    }
    training_params = {
        "batch_size": 48,
        "buffer_size": 50000,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "initial_epsilon" : 0.5,
        "final_epsilon" : 1e-3,
        "teacher_update_freq" : 10,
        "report_interval" : 10,
        "episode_count" : 1500,
        "initial_count" : 50,
        "observe_interval" : 100,
        "log_dir": ""
    }
    dataset_params = {
        "sequence_length": 48,
        "slicing_size": 1
    }

    # それぞれのモデルパラメータ候補を辞書に格納する
    model_params = model_params_candinate(model_params)

    wandb.login(key = WANDB_API_KEY)

    env = gym.make("CartPole-v1")

    try:
        for each_model_params in model_params:

            for trial in range(trial_number):

                obs = CartPoleObserver(env)

                input_size = env.observation_space.shape[0]
                output_size = env.action_space.n

                each_model_params.update({"input_size": input_size,"ReadOut_output_size": output_size})

                file_name = "esn_agent.pth" if not is_test else "esn_agent_test.pth"

                optimizer = "Adam"
                criterion = "SmoothL1Loss"

                # モデルとデータセットのパラメータを定義
                training_params.update({"file_name": file_name})

                trainer = ESNTrainer(model_params=each_model_params, dataset_params=dataset_params, training_params=training_params, optimizer=optimizer, criterion=criterion)
                path = trainer.logger.path_of(trainer.file_name)
                agent_class = ESNAgent

                memory = RecurrentExperienceReplayMemory(training_params["buffer_size"], dataset_params["sequence_length"])

                # wandb.init inside loop with try-finally to ensure wandb.finish() is called
                try:
                    wandb.init(project="RC_Touritsu", config = {
                        "env": obs.__class__.__name__,
                        "architecture": "ESN",
                        "model_params" : each_model_params,
                        "training_params" : training_params,
                        "memory" : memory.__class__.__name__,
                        "criterion" : criterion,
                        "optimizer" : optimizer,
                    }, reinit=True
                    )

                    if play:
                        print("play")

                        # モデルのロード
                        model = ESN(each_model_params, training_params, dataset_params).to(device)
                        model.load_state_dict(torch.load(path, map_location=device))
                        model.ReadOut.eval()  # 推論モードに設定

                        # エージェントを全てのパラメータで初期化
                        agent = ESNAgent(
                            epsilon=0.0,
                            actions=list(range(output_size)),
                            model_params=each_model_params,
                            training_params=training_params,  # 修正: training_params を追加
                            dataset_params=dataset_params,
                            memory=memory
                        )
                        agent.model = model
                        agent.play(env, render=True)
                    else:
                        print("train")
                        trainer.train(env, memory, test_mode=is_test)

                except KeyboardInterrupt:
                    print("Interrupted by user. Finishing WandB run.")
                    wandb.finish()
                    raise  # 再度例外を投げて外側でも処理させる
                finally:
                    wandb.finish()  # 例外が発生してもしなくても必ず終了処理

    except KeyboardInterrupt:
        print("Training interrupted by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNN Agent for CartPole")
    parser.add_argument("--play", action="store_true",
                        help="trained modelでプレイする")
    parser.add_argument("--test", action="store_true",
                        help="テストモードで学習する")

    args = parser.parse_args()
    main(args.play, args.test)
