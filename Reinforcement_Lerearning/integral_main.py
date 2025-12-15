import re
import os
import time
import random
import argparse
import itertools
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

from dotenv import load_dotenv
from collections import namedtuple, deque
from multiprocessing import Process, Queue

from ESN import ESN
# from ESN_Agent import ESNAgent, ESNTrainer
from ESN_Agent_episode_equalstep import ESNAgent, ESNTrainer

from Replay_Memory import RecurrentExperienceReplayMemory
from fn_framework_torch import FNAgent, Trainer, Observer

# from Environment import TurningUpControlEnv, sensor_process
from Environment_timestamp import TurningUpControlEnv, sensor_process

import wandb

load_dotenv()

WANDB_API_KEY = os.getenv('WANDB_KEY')  # あなたのWandB APIキーをここに入力してください

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import logging
import sys

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

def main(play, is_test, time_logger):
    trial_number = 1
    seed_number = 43
    set_seed(seed_number)

    model_params = {
        "reservoir_size" : [100],
        "channel_size" : [1],
        "reservoir_weights_scale" : [1],
        "input_weights_scale" : [1],
        "spectral_radius" : [0.9],
        "reservoir_density" : [0.02],
        "reak_rate" : [0.5],
        "Batch_Training" : [False]
    }
    training_params = {
        "batch_size": 1,
        "buffer_size": 10,
        "learning_rate": 0.01,
        "gamma": 0.99,
        "initial_epsilon" : 0.5,
        "final_epsilon" : 1e-3,
        "teacher_update_freq" : 10,
        "report_interval" : 5,
        "episode_count" : 3,
        "initial_count" : 1,
        "observe_interval" : 1,
        "process_wait_time" : 5,
        "log_dir": "",
        "max_steps_per_episode": 25,  # 必要に応じて値を調整
        "test_count" : 5
    }

    environment_params = {
        "state_length": 250,#状態行列(RCモデルへの入力行列)の時系列長．この値を元に，触覚センサ値をダウンサンプリングする．
        "reward_length": 50,#報酬ベクトルの時系列長．

        "reward_addition": 5,#Z軸負の方向へヘッダを移動させることを目的とした，rewardの項
        "reward_penalty": -20,#ヘッダの移動制限を超えた場合における，rewardの項

        "move_X": 60,#基準となるX軸移動の速度と座標
        "move_Velocity": 1200,
        
        "init_Z": 10,
        "MIN_X": 20,# 例: X軸の最小位置
        "MAX_X": 100,# 例: X軸の最大位置（必要なら）

        "MIN_Z": 5,# 例: Z軸の最小位置
        "MAX_Z": 10,# 例: Z軸の最大位置（必要なら）

        "inv_distance" : 1e-3,
        "threshold_distance" : 10,

        "material" : "trasing paper"
    }

    #datasetじゃなくてバッファの設定かも
    dataset_params = {
        "sequence_length": 10,
        "slicing_size": 1
    }

    actions = actions = ["G1 Z0.01 F9000","G1 Z-0.01 F9000", "-10", "10", "G1 Z0 F9000"]

    # それぞれのモデルパラメータ候補を辞書に格納する
    model_params = model_params_candinate(model_params)

    wandb.login(key = WANDB_API_KEY)

    # キューの作成
    state_queue = Queue()
    reward_queue = Queue()
    
    # センサープロセスの開始
    sensor_proc = Process(target=sensor_process, args=(state_queue, reward_queue))
    sensor_proc.start()

    try:
        for each_model_params in model_params:

            for trial in range(trial_number):
                env = TurningUpControlEnv(state_queue, reward_queue, actions, environment_params, time_logger)

                input_size = env.observation_space.shape[1]
                output_size = env.action_space.n

                each_model_params.update({"input_size": input_size,"ReadOut_output_size": output_size})

                file_name = "esn_agent.pth" if not is_test else "esn_agent_test.pth"

                optimizer = "Adam"
                criterion = "SmoothL1Loss"

                # モデルとデータセットのパラメータを定義
                training_params.update({"file_name": file_name})

                trainer = ESNTrainer(each_model_params, dataset_params, training_params, optimizer, criterion, time_logger)
                path = trainer.logger.path_of(trainer.file_name)
                agent_class = ESNAgent

                memory = RecurrentExperienceReplayMemory(training_params["buffer_size"], dataset_params["sequence_length"])

                # wandb.init inside loop with try-finally to ensure wandb.finish() is called
                try:
                    wandb.init(project="Python_RL_Torch", config = {
                        "env": env.__class__.__name__,
                        "architecture": "ESN",
                        "model_params" : each_model_params,
                        "training_params" : training_params,
                        "environment_params" : environment_params,
                        "dataset_params" : dataset_params,
                        "memory" : memory.__class__.__name__,
                        "criterion" : criterion,
                        "optimizer" : optimizer,
                        "actions" : actions,
                        "seed_number" : seed_number
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

                    # プロセスの終了処理
                    sensor_proc.terminate()
                    sensor_proc.join()
                    env.printer_serial_close()
                    print("Training finished and resources cleaned up.")

    except KeyboardInterrupt:
        print("Training interrupted by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNN Agent for CartPole")
    parser.add_argument("--play", action="store_true",
                        help="trained modelでプレイする")
    parser.add_argument("--test", action="store_true",
                        help="テストモードで学習する")

    # すでに logging.basicConfig(...) などで設定済みの部分があれば調整してください
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,          # INFO以上のレベルのログを表示
        format=LOG_FORMAT,
        stream=sys.stdout            # 標準出力へログを表示
    )
    time_logger = logging.getLogger(__name__)  # モジュール全体で使うロガー

    args = parser.parse_args()
    main(args.play, args.test, time_logger)
