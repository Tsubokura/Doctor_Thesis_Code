import re
import time
import random
import argparse
import itertools
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from PIL import Image
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim

from ESN import ESN
from fn_framework_torch import FNAgent, Trainer, Observer

import logging
import sys

class Timer:
    """
    with ブロックの開始時刻と終了時刻、経過時間を自動的にログ出力するコンテキストマネージャ
    """
    def __init__(self, msg=""):
        self.msg = msg

    def __enter__(self):
        self.start_time = time.time()
        logging.info(f"[START] {self.msg}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        elapsed = end_time - self.start_time
        logging.info(f"[END] {self.msg} - elapsed={elapsed:.3f} sec")


#1ステップ，つまり3Dプリンターが行う一回の移動と，1エピソードが同じ時間という設定にて強化学習を実施．

Experience = namedtuple("Experience", ["s", "a", "r", "n_s", "d"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ESNAgent(FNAgent):

    def __init__(self, epsilon, actions, model_params, training_params, dataset_params, memory, optimizer, criterion, time_logger):
        super().__init__(epsilon, actions)
        self.model_params = model_params
        self.training_params = training_params
        self.dataset_params = dataset_params

        self.memory = memory
        
        self.gamma = float(training_params["gamma"])
        self.lr = float(training_params["learning_rate"])
        self.capacity = int(training_params["buffer_size"])
        self.batch_size = int(training_params["batch_size"])
        self.sequence_length = int(dataset_params["sequence_length"])

        # ESNの初期化
        self.model = ESN(model_params, training_params, dataset_params).to(device)
        self.model.ReadOut.to(device)
        self.model.ReadOut.train()  # ReadOut層を訓練モードに設定

        # ターゲットモデルの初期化（安定性のため）
        self.target_model = ESN(model_params, training_params, dataset_params).to(device)
        self.target_model.ReadOut.eval()
        self.target_model.ReadOut.load_state_dict(self.model.ReadOut.state_dict())

        self.time_logger = time_logger
        

        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        if criterion == "SmoothL1Loss":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")

    def initialize(self, experiences):
        self.initialized = True
        print("Agent initialized.")

    def reset_hidden_state(self):
        # ESNは内部でリザバー状態を管理するため、特別なリセットは不要
        self.model.reset_hidden_state()

    def estimate(self, sequences):
        # sequences: シーケンスのリスト、各シーケンスはExperienceのリスト
        states = np.array([[exp.s for exp in seq] for seq in sequences], dtype=np.float32)
        states = torch.tensor(states, dtype=torch.float32, device=device)

        outputs = self.model(states)
        # outputsの形状: (バッチサイズ, シーケンス長, 出力サイズ)
        # 最後のタイムステップの出力を取得
        last_outputs = outputs[:, -1, :]
        return last_outputs  # 形状: (バッチサイズ, 出力サイズ)

    def update(self):
        self.time_logger.info("----------------Agent_model_update_process_start----------------\n")
        # print("update memory : ", len(self.memory))
        
        if len(self.memory) < self.memory.seq_length * self.batch_size:
            print("training data amount is insufficient")
            return  # 学習に十分なデータがない場合
        sequences = self.memory.sample(self.batch_size)
        
        if not sequences:
            return

        states = np.array([[exp.s for exp in seq] for seq in sequences], dtype=np.float32)
        actions = np.array([[exp.a for exp in seq] for seq in sequences], dtype=np.float32)
        rewards = np.array([[exp.r for exp in seq] for seq in sequences], dtype=np.float32)
        next_states = np.array([[exp.n_s for exp in seq] for seq in sequences], dtype=np.float32)
        dones = np.array([[exp.d for exp in seq] for seq in sequences], dtype=np.float32)

        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        batch_size, channel_size, input_sequence_length, input_size = states.shape
        input_new_sequence_length = channel_size *input_sequence_length
        states = states.reshape(batch_size, 1, input_new_sequence_length, input_size)
        next_states = next_states.reshape(batch_size, 1, input_new_sequence_length, input_size)

        batch_size, channel_size, reward_sequence_length = rewards.shape
        reward_ew_sequence_length = channel_size * reward_sequence_length
        rewards = rewards.reshape(batch_size, 1, reward_ew_sequence_length)


        # print("After torch.tensor states shape:", states.shape)

        # 現在の状態の予測
        self.model.train()
        self.optimizer.zero_grad()

        # outputsの形状を調整
        # print(f"state {states.shape }")
        # print(f"action {actions.shape }")
        outputs = self.model(states)
        # outputs: [バッチサイズ, シーケンス長, 出力サイズ]
        # 最終タイムステップの出力を取得
        outputs = outputs[:, -1, :]  # [バッチサイズ, 出力サイズ]
        

        # Q値の取得
        q_values = outputs.gather(1, actions[:, -1].unsqueeze(1)).squeeze(1)
        # print(f"rewards in Qvalue proc {rewards}")
        # print(f"rewards in Qvalue proc shape {rewards.shape}")

        # ターゲットQ値の計算
        with torch.no_grad():
            next_outputs = self.target_model(next_states)
            next_outputs = next_outputs[:, -1, :]  # [バッチサイズ, 出力サイズ]
            next_q_values = next_outputs.max(1)[0]
            # final_rewards = rewards[:, :, -1].squeeze(1)
            final_rewards = rewards[:, :, -50:].mean(dim=2).squeeze(1)
            # print(f"reward in Qvalue proc {final_rewards}")
            target_q_values = final_rewards + (self.gamma * next_q_values * (1 - dones[:, -1]))

        loss = self.loss_fn(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.time_logger.info("----------------Agent_model_update_process_end----------------\n")
        
        # 損失を返す
        return loss.item()

    def update_teacher(self):
        # ターゲットReadOutの更新
        self.target_model.ReadOut.load_state_dict(self.model.ReadOut.state_dict())

    def policy(self, s):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            s = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, 入力サイズ)
            with torch.no_grad():
                # print("policy:", s.shape)
                outputs = self.model(s)
                q_values = outputs.squeeze(0).squeeze(0)  # 形状: (出力サイズ)
                # print("q_ : ", q_values)
                # print("q_shape : ", q_values.shape)
                #最終時の出力Q値のみ用いる
                action = q_values[-1, :].argmax().item()
            return action

    def play(self, env, episode_count=5, render=True):
        for e in range(episode_count):
            s = env.reset()
            self.reset_hidden_state()  # 各エピソードの開始時に隠れ状態をリセット
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, done, _, info = env.step(a)
                episode_reward += reward
                s = n_state
            else:
                print("リワードを獲得: {}.".format(episode_reward))


class ESNTrainer(Trainer):
    def __init__(self, model_params, dataset_params, training_params, optimizer, criterion, time_logger):

        self.optimizer = optimizer
        self.criterion = criterion

        self.model_params = model_params
        self.dataset_params = dataset_params
        self.training_params = training_params
        
        self.sequence_length = int(dataset_params["sequence_length"])
        self.buffer_size = int(training_params["buffer_size"])
        self.batch_size = int(training_params["batch_size"])

        self.gamma = float(training_params["gamma"])
        self.initial_epsilon = float(training_params["initial_epsilon"])
        self.final_epsilon = float(training_params["final_epsilon"])
        self.learning_rate = float(training_params["learning_rate"])
        self.episode_count = int(training_params["episode_count"])
        self.initial_count = int(training_params["initial_count"])
        self.teacher_update_freq = int(training_params["teacher_update_freq"])
        self.observe_interval = int(training_params["observe_interval"])
        self.max_steps_per_episode = int(training_params.get("max_steps_per_episode", 500))  # デフォルト値は500

        self.loss = 0
        self.training_episode = 0
        self._max_reward = -10

        self.process_wait_time = float(training_params["process_wait_time"])
        self.test_count = int(training_params["test_count"])
        self.file_name = training_params.get("file_name", "esn_agent.pth")  # デフォルト値を設定
        log_dir = training_params.get("log_dir", "")

        # 最大ステップ数を読み込む
        self.report_interval = int(training_params["report_interval"])

        self.env = None

        self.time_logger = time_logger

        super().__init__(self.buffer_size, self.batch_size, self.gamma, self.report_interval, log_dir)

    def train(self, env, memory, test_mode=False, render=False):
        self.env = env
        actions = list(range(env.action_space.n))
        
        agent = ESNAgent(
            epsilon=self.initial_epsilon,
            actions=actions,
            model_params=self.model_params,
            training_params=self.training_params,
            dataset_params=self.dataset_params,
            memory = memory,
            optimizer = self.optimizer,
            criterion = self.criterion,
            time_logger = self.time_logger
        )
        self.training_episode = self.episode_count

        self.train_loop(env, agent, self.episode_count, self.initial_count, self.observe_interval)
        return agent

    def train_loop(self, env, agent, episode=200, initial_count=-1, observe_interval=0, render=False):
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        frames = []
        self.time_logger.info("------------------Trainer_train_loop----------------\n")

        wandb_step_count = 0

        env.init_procedure()
        time.sleep(10)

        env.reset()
        s = np.zeros((250, 3))#状態行列の初期値．時系列長は，触覚センサのサンプリングレート×3秒=(1500)から1/10ダウンサンプリングすることによる決め打ち
        # print("reset :", s.shape)

        # env.first_move()

        # time.sleep(4)


        for i in range(episode):
            self.time_logger.info("------------------Trainer_episode_init----------------\n")
            
            # エピソードごとの初期処理
            agent.reset_hidden_state()
            env.reset()
            done = False
            self.total_reward = 0
            self.step_loss = 0

            step_count = 0

            self.episode_begin(i, agent)
            while not done:
                self.time_logger.info("-----------------Trainer_episode_loop_start----------------\n")

                start_time = time.time()
                if render:
                    env.render()
                if self.training and observe_interval > 0 and \
                        (self.training_count == 1 or
                         self.training_count % observe_interval == 0):
                    frames.append(s)

                a = agent.policy(s)
                # print("action", a)
                with Timer("step_episode_equalstep"):
                    n_state, reward, done, info = env.step_episode_equalstep(a)
                # print("done", done)
                # print(f"Next State: {n_state.shape}")
                # print(f"Reward: {reward}")
                # print(f"Reward shape: {reward.shape}")

                e = Experience(s, a, reward, n_state, done)
                agent.memory.push(e)
                # print("done", done)

                # print("memory:", len(agent.memory))
                if not self.training and \
                        len(agent.memory) >= self.buffer_size:
                    self.time_logger.info("Training開始\n")
                    self.begin_train(i, agent)
                    self.training = True

                # モデルの更新処理
                with Timer("step and update process"):
                    self.step(i, step_count, wandb_step_count, agent, e)

                # print(f"All Process End_time {time_difference}")

                s = n_state

                # ステップ数が最大を超えた場合、エピソードを終了
                if step_count >= self.max_steps_per_episode:
                    print(f"Episode {i} が最大ステップ数 ({self.max_steps_per_episode}) を超えました。エピソードを終了します。")
                    done = True
                    # オプション: ペナルティを追加
                    # self.total_reward -= 10  # 例: ステップ数超過によるペナルティ

                # 報酬の処理
                

                # print(f"rewards in step {reward}")
                # print(f"rewards shape in step {reward.shape}")
                if isinstance(reward, torch.Tensor):
                    reward = reward.mean().item()
                elif isinstance(reward, np.ndarray):
                    reward = reward.mean()
                elif isinstance(reward, (list, tuple)):
                    reward = np.mean(reward)

                # print(f"rewards in step {reward}")
                
                
                # print(reward.shape)
                # if isinstance(reward, np.ndarray):
                #     # reward.shape は (seq_length,) のはずなので、0番目の軸がシーケンス長
                #     seq_length = reward.shape[0]
                #     half_index = seq_length // 2
                #     # 後半部分の平均値を算出（1D 配列なので axis を指定する必要はありません）
                #     reward = reward[half_index:].mean()
                # elif isinstance(reward, torch.Tensor):
                #     # torch.Tensor の場合も同様に 0 番目の軸がシーケンス長
                #     seq_length = reward.size(0)
                #     half_index = seq_length // 2
                #     reward = reward[half_index:].mean()
            

                self.total_reward += reward

                interval_time = time.time()

                env.move_to_first_position()

                time_difference = interval_time - start_time

                if self.process_wait_time > time_difference:
                    self.time_logger.info("---------------Trainer_process_wait_time----------------\n")
                    print(f"process_wait_time : {self.process_wait_time - time_difference}")
                    time.sleep(self.process_wait_time - time_difference)
                else:
                    print("process time is over, please check")


                step_count += 1
                wandb_step_count += 1

                self.time_logger.info("---------------Trainer_episode_loop_end----------------\n")

            
            # エピソード終了
            env.move_to_initial_position() #episode開始時の時点まで戻す．
            self.reward_log.append(self.total_reward)
            self.episode_end(i, step_count, agent)

            if not self.training and \
                    initial_count > 0 and i >= initial_count:
                self.begin_train(i, agent)
                self.training = True

            if self.training:
                if len(frames) > 0:
                    # self.logger.write_image(self.training_count, frames)
                    frames = []
                self.training_count += 1

            end_time = time.time()
            elapsed_time = end_time - start_time
            # print(f"処理にかかった時間: {elapsed_time} 秒")

            self.time_logger.info("----------------Trainer_episode_end----------------\n")
        
        #すべてのepisodeが終了したあと，テストを行う．
        time.sleep(0.5)
        self.auto_test(env, agent, self.test_count)

    def episode_begin(self, episode, agent):
        self.loss = 0

    def begin_train(self, episode, agent):
        agent.initialize(self.experiences)
        self.logger.set_model(agent.model)
        agent.epsilon = self.initial_epsilon
        self.training_episode -= episode

    def step(self, episode, step_count, wandb_step_count, agent, experience):
        self.time_logger.info("------------------Trainer_step_and_update_process----------------\n")

        #1ステップごとに，wandbへ記録．
        self.logger.write(wandb_step_count, "loss", self.step_loss)
        self.logger.write(wandb_step_count, "reward", experience.r[-1])
        self.logger.write(wandb_step_count, "epsilon", agent.epsilon)
        self.logger.write(wandb_step_count, "total_reward", self.total_reward)
        self.logger.write(wandb_step_count, "total_loss", self.loss)

        if self.training:
            self.step_loss = agent.update()

            # print(type(self.loss))
            # print(type(self.step_loss))
            if self.loss is not None and self.step_loss is not None:
                self.loss += self.step_loss

    def episode_end(self, episode, step_count, agent):
        print("episode_end")
        reward = self.reward_log[-1]
        if step_count > 0:
            self.loss = self.loss / step_count
        else:
            self.loss = 0

        # if training
        # self.logger.write(self.training_count, "loss", self.loss)
        # self.logger.write(self.training_count, "reward", reward)
        # self.logger.write(self.training_count, "epsilon", agent.epsilon)
        if reward > self._max_reward:
            torch.save(agent.model.state_dict(), self.logger.path_of(self.file_name))
            self._max_reward = reward
        # if self.is_event(self.training_count, self.teacher_update_freq):
        #     agent.update_teacher()
        agent.update_teacher()

        diff = (self.initial_epsilon - self.final_epsilon)
        decay = diff / self.training_episode
        agent.epsilon = max(agent.epsilon - decay, self.final_epsilon)

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)

        

    def auto_test(self, env, agent, test_steps=30):
        """
        一定のエピソード数が経過したり、一定の報酬水準を達成した際に自動的に呼び出されるテスト用メソッド。
        """
        print("\n---------- 自動テスト開始 ----------")
        if self.env is None:
            print("Trainerに環境が設定されていません。self.env がNoneです。")
            return

        # テスト中は探索しない（ε=0）ため、元の値を避難しておく
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0

        test_rewards = []
        env.reset()
        env.iftest = True
        agent.reset_hidden_state()

        time.sleep(5)
        s = np.zeros((250, 3))
        for i in range(test_steps):
            start_time = time.time()
            
            a = agent.policy(s)
            s, reward, done, info = env.step_episode_equalstep(a)
            # print(s)
            # if a == 1:
            #     reward -= self.env.reward_addition

            test_rewards.append(reward)
            print(f"[Test Step {i+1}] リワード: {reward}")

            interval_time = time.time()
            env.move_to_first_position()

            time_difference = interval_time - start_time

            if self.process_wait_time > time_difference:
                print(self.process_wait_time - time_difference)
                time.sleep(self.process_wait_time - time_difference)
            else:
                print("process time is over, please check")

        avg_reward = sum(test_rewards) / len(test_rewards)
        print(f"【平均テストリワード】: {avg_reward}")
        print("---------- 自動テスト終了 ----------\n")

        # テスト終了後に ε を元に戻す
        agent.epsilon = original_epsilon

