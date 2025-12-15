
import time
import re
import random
import argparse
import itertools
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import gymnasium as gym
# import gym_ple
from fn_framework_torch import FNAgent, Trainer, Observer, Logger
from ESN import ESN

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import wandb

#バッチサイズの処理を修正してみる

Experience = namedtuple("Experience", ["s", "a", "r", "n_s", "d"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ESNAgent(FNAgent):

    def __init__(self, epsilon, actions, model_params, training_params, dataset_params, memory, optimizer, criterion):
        super().__init__(epsilon, actions)
        self.model_params = model_params
        self.training_params = training_params
        self.dataset_params = dataset_params

        self.sequence_length = int(dataset_params["sequence_length"])
        
        self.capacity = int(training_params["buffer_size"])
        self.lr = float(training_params["learning_rate"])
        self.gamma = float(training_params["gamma"])

        # ESNの初期化
        self.model = ESN(model_params, training_params, dataset_params).to(device)
        self.model.ReadOut.to(device)
        self.model.ReadOut.train()  # ReadOut層を訓練モードに設定

        # ターゲットモデルの初期化（安定性のため）
        self.target_model = ESN(model_params, training_params, dataset_params).to(device)
        self.target_model.ReadOut.load_state_dict(self.model.ReadOut.state_dict())
        self.target_model.ReadOut.eval()

        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        else:
            None#エラー文処理をしろ
        
        if criterion == "SmoothL1Loss":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            None

        self.memory = memory
        
        
        self.batch_size = int(training_params["batch_size"])

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
        if len(self.memory) < self.memory.seq_length * self.batch_size:
            return None, None, None # 学習に十分なデータがない場合
        sequences = self.memory.sample(self.batch_size)
        
        if not sequences:
            return

        states = np.array([[exp.s for exp in seq] for seq in sequences], dtype=np.float32)
        actions = np.array([[exp.a for exp in seq] for seq in sequences], dtype=np.float32)
        rewards = np.array([[exp.r for exp in seq] for seq in sequences], dtype=np.float32)
        next_states = np.array([[exp.n_s for exp in seq] for seq in sequences], dtype=np.float32)

        for seq in sequences:
            assert all(not exp.d for exp in seq[:-1]), "Sequence crosses episode boundary."

        dones = np.array([[exp.d for exp in seq] for seq in sequences], dtype=np.float32)

        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        states = states.unsqueeze(1)

        next_states = next_states.unsqueeze(1)


        # 現在の状態の予測
        self.model.train()
        self.optimizer.zero_grad()


        # outputsの形状を調整
        outputs = self.model(states)
        # outputs: [バッチサイズ, シーケンス長, 出力サイズ]
        # 最終タイムステップの出力を取得
        outputs = outputs[:, -1, :]  # [バッチサイズ, 出力サイズ]

        # Q値の取得
        q_values = outputs.gather(1, actions[:, -1].unsqueeze(1)).squeeze(1)
        # print(f"q_values  update {q_values }")

        model_proc_start_time = time.time()
        # ターゲットQ値の計算
        with torch.no_grad():
            next_outputs = self.target_model(next_states)
            next_outputs = next_outputs[:, -1, :]  # [バッチサイズ, 出力サイズ]
            next_q_values = next_outputs.max(1)[0]
            target_q_values = rewards[:, -1] + (self.gamma * next_q_values * (1 - dones[:, -1]))

        loss = self.loss_fn(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        model_proc_end_time = time.time()
        model_proc_time = model_proc_end_time - model_proc_start_time

        # print(states.shape)
        # 損失を返す
        return loss.item(), states.shape, model_proc_time

    def update_teacher(self):
        # ターゲットReadOutの更新
        self.target_model.ReadOut.load_state_dict(self.model.ReadOut.state_dict())

    def policy(self, s):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            s = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, 入力サイズ)
            with torch.no_grad():
                outputs = self.model(s)
                q_values = outputs.squeeze(0).squeeze(0)  # 形状: (出力サイズ)
                action = q_values.argmax().item()
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
    def __init__(self, model_params, dataset_params, training_params, optimizer, criterion):
        buffer_size = int(training_params["buffer_size"])
        batch_size = int(training_params["batch_size"])
        gamma = float(training_params["gamma"])
        report_interval = int(training_params["report_interval"])
        log_dir = training_params.get("log_dir", "")
        file_name = training_params.get("file_name", "esn_agent.pth")  # デフォルト値を設定
        super().__init__(buffer_size, batch_size, gamma,
                         report_interval, log_dir)

        self.model_params = model_params
        self.dataset_params = dataset_params
        self.training_params = training_params

        self.sequence_length = int(dataset_params["sequence_length"])
        self.file_name = file_name if file_name else "esn_agent.pth"
        self.initial_epsilon = float(training_params["initial_epsilon"])
        self.final_epsilon = float(training_params["final_epsilon"])
        self.learning_rate = float(training_params["learning_rate"])
        self.teacher_update_freq = int(training_params["teacher_update_freq"])
        self.episode_count = int(training_params["episode_count"])
        self.initial_count = int(training_params["initial_count"])
        self.observe_interval = int(training_params["observe_interval"])
        self.loss = 0
        self.training_episode = 0
        self._max_reward = -10

        self.optimizer = optimizer
        self.criterion = criterion

        self.wandb_step_number = 0

        # 時間計測データを保存するリストを初期化
        self.step_elapsed_times = []
        self.model_proc_elapsed_times = []

        self.histogram_plot_timing = 10000
    

    def train(self, env, memory, test_mode=False, render=False):
        actions = list(range(env.action_space.n))
        
        agent = ESNAgent(
            epsilon=self.initial_epsilon,
            actions=actions,
            model_params=self.model_params,
            training_params=self.training_params,
            dataset_params=self.dataset_params,
            memory = memory,
            optimizer = self.optimizer,
            criterion = self.criterion
        )
        self.training_episode = self.episode_count

        self.train_loop(env, agent, self.episode_count, self.initial_count, self.observe_interval)
        return agent

    def train_loop(self, env, agent, episode=200, initial_count=-1, observe_interval=0, render=False, ):
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        frames = []
        
        for i in range(episode):
            start_time = time.time()
            s = env.reset()[0]

            agent.reset_hidden_state()  # 各エピソードの開始時にリザバー状態をリセット
            done = False
            step_count = 0
            total_reward = 0
            self.episode_begin(i, agent)
            while not done:
                step_start_time = time.time()
                if render:
                    env.render()
                if self.training and observe_interval > 0 and \
                        (self.training_count == 1 or
                         self.training_count % observe_interval == 0):
                    frames.append(s)

                # print(f"s shape {s.shape}")
                a = agent.policy(s)
                n_state, reward, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                # n_state = n_state.reshape(n_state.shape[0], 1)
                e = Experience(s, a, reward, n_state, done)
                
                agent.memory.push(e)

                if not self.training and \
                        len(agent.memory) >= self.buffer_size:
                    self.begin_train(i, agent)
                    self.training = True

                self.step(i, step_count, agent, e)

                s = n_state
                step_count += 1
                total_reward += reward
                step_end_time = time.time()
                step_elapsed_time = step_end_time - step_start_time
                

                # print(f"第{step} ステップの処理にかかった時間: {step_elapsed_time} 秒")
                self.logger.write(self.wandb_step_number, "step_elapsed_time", step_elapsed_time)

                self.step_elapsed_times.append(step_elapsed_time)
                # wandb.log({"step_elapsed_time_histogram": wandb.Histogram(step_elapsed_time), "step": self.wandb_step_number})
                # self.logger.write(self.wandb_step_number, "length", s.shape[0])

                if self.wandb_step_number % self.histogram_plot_timing == 0 and self.wandb_step_number > 0:
                    # self.logger.plot_histogram(self.step_elapsed_times, "step_elapsed_time")
                    self.logger.summarize_times(self.step_elapsed_times, "step_elapsed_times Distribution")
             
                self.wandb_step_number += 1
            else:
                # print(f"reward {total_reward}")
                self.reward_log.append(total_reward)
                self.episode_end(i, step_count, agent)

                if not self.training and \
                        initial_count > 0 and i >= initial_count:
                    self.begin_train(i, agent)
                    self.training = True

                if self.training:
                    if len(frames) > 0:
                        self.logger.write_image(self.training_count,
                                                frames)
                        frames = []
                    self.training_count += 1
                
            end_time = time.time()
            episode_elapsed_time = end_time - start_time
            # self.logger.write("episode_elapsed_time", episode_elapsed_time)
            

    def episode_begin(self, episode, agent):
        self.loss = 0

    def begin_train(self, episode, agent):
        agent.initialize(self.experiences)
        self.logger.set_model(agent.model)
        agent.epsilon = self.initial_epsilon
        self.training_episode -= episode

    def step(self, episode, step_count, agent, experience):
        if self.training:
            # loss, state_length, model_proc_time = agent.update()
            # self.logger.write(self.wandb_step_number, "state_length", state_length)
            # self.logger.write(self.wandb_step_number, "model_proc_time", model_proc_time)
            # if loss is not None:
            #     self.loss += loss
            result = agent.update()
            if result is not None and result[0] is not None and result[1] is not None and result[2] is not None:
                loss, state_length, model_proc_time = result
                self.loss += loss
                # self.logger.write(self.wandb_step_number, "loss", loss)
                self.logger.write(self.wandb_step_number, "state_length", state_length[2])
                self.logger.write(self.wandb_step_number, "state_batch", state_length[0])
                self.logger.write(self.wandb_step_number, "model_proc_time", model_proc_time)

                self.model_proc_elapsed_times.append(model_proc_time)

                if self.wandb_step_number % self.histogram_plot_timing == 0 and self.wandb_step_number > 0:
                    # self.logger.plot_histogram(self.model_proc_elapsed_times, "model_proc_elapsed_times")
                    self.logger.summarize_times(self.model_proc_elapsed_times, "model_proc_elapsed_times Distribution")

                # wandb.log({"model_proc_time_histogram": wandb.Histogram(model_proc_time), "step": self.wandb_step_number})
               

    def episode_end(self, episode, step_count, agent):
        reward = self.reward_log[-1]
        if step_count > 0:
            self.loss = self.loss / step_count
        else:
            self.loss = 0

        if self.training:
            # self.logger.write(self.training_count, "loss", self.loss)
            # self.logger.write(self.training_count, "reward", reward)
            # self.logger.write(self.training_count, "epsilon", agent.epsilon)

            if reward > self._max_reward:
                torch.save(agent.model.state_dict(), self.logger.path_of(self.file_name))
                self._max_reward = reward
            if self.is_event(self.training_count, self.teacher_update_freq):
                agent.update_teacher()

            diff = (self.initial_epsilon - self.final_epsilon)
            decay = diff / self.training_episode
            agent.epsilon = max(agent.epsilon - decay, self.final_epsilon)

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)

