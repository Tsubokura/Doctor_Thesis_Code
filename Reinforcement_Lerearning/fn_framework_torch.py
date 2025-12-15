import os
import io
import re
import wandb
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt


Experience = namedtuple("Experience", ["s", "a", "r", "n_s", "d"])


class FNAgent():

    def __init__(self, epsilon, actions, model=None):
        self.epsilon = epsilon
        self.actions = actions
        self.model = model
        self.estimate_probs = False
        self.initialized = model is not None

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions)
        agent.model = cls.create_model()  # Assuming a method to create the model
        agent.model.load_state_dict(torch.load(model_path))
        agent.initialized = True
        return agent

    def initialize(self, experiences):
        raise NotImplementedError("You have to implement initialize method.")

    def estimate(self, s):
        raise NotImplementedError("You have to implement estimate method.")

    def update(self, experiences, gamma):
        raise NotImplementedError("You have to implement update method.")

    def policy(self, s):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            estimates = self.estimate(s)
            if self.estimate_probs:
                action = np.random.choice(self.actions, size=1, p=estimates)[0]
                return action
            else:
                return torch.argmax(torch.tensor(estimates)).item()

    def play(self, env, episode_count=5, render=True):
        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                s = n_state
            else:
                print("Get reward {}.".format(episode_reward))


class Trainer():

    def __init__(self, buffer_size=1024, batch_size=32,
                 gamma=0.9, report_interval=10, log_dir=""):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.report_interval = report_interval
        self.logger = Logger(log_dir, self.trainer_name)
        self.experiences = deque(maxlen=buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []

    @property
    def trainer_name(self):
        class_name = self.__class__.__name__
        snaked = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        snaked = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snaked).lower()
        snaked = snaked.replace("_trainer", "")
        return snaked

    def train_loop(self, env, agent, episode=200, initial_count=-1,
                   render=False, observe_interval=0):
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        frames = []

        for i in range(episode):
            # print("reset")
            s = env.reset()
            done = False
            step_count = 0
            # print("reset done")
            self.episode_begin(i, agent)
            while not done:
                if render:
                    env.render()
                if self.training and observe_interval > 0 and\
                   (self.training_count == 1 or
                    self.training_count % observe_interval == 0):
                    frames.append(s)

                a = agent.policy(s)
                # print("env step")
                n_state, reward, done, info = env.step(a)
                e = Experience(s, a, reward, n_state, done)
                # print(e)
                self.experiences.append(e)
                if not self.training and \
                   len(self.experiences) == self.buffer_size:
                    self.begin_train(i, agent)
                    self.training = True

                # print("self step")
                self.step(i, step_count, agent, e)

                s = n_state
                step_count += 1
            else:
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

    def episode_begin(self, episode, agent):
        pass

    def begin_train(self, episode, agent):
        pass

    def step(self, episode, step_count, agent, experience):
        pass

    def episode_end(self, episode, step_count, agent):
        pass

    def is_event(self, count, interval):
        return True if count != 0 and count % interval == 0 else False

    def get_recent(self, count):
        recent = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent]


class Observer():

    def __init__(self, env):
        self._env = env

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        # print("transform", self.transform(self._env.reset())[0])

        return self.transform(self._env.reset()[0])

    def render(self):
        self._env.render(mode="human")

    def step(self, action):
        results = self._env.step(action)
        # print("Step results:", results)
        # print("Step results:", results[0])
        n_state, reward, done, truncated, info = results
        done = done or truncated
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        raise NotImplementedError("You have to implement transform method.")

class Logger():

    def __init__(self, log_dir="", dir_name="", project_name="pytorch_project"):
        self.log_dir = log_dir
        if not log_dir:
            self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if dir_name:
            self.log_dir = os.path.join(self.log_dir, dir_name)
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)


    def set_model(self, model):
        # WandB でモデルを監視（パラメータや勾配を追跡）
        wandb.watch(model, log="all")

    def path_of(self, file_name):
        return os.path.join(self.log_dir, file_name)

    def describe(self, name, values, episode=-1, step=-1):
        mean = np.round(np.mean(values), 3)
        std = np.round(np.std(values), 3)
        desc = "{} is {} (+/-{})".format(name, mean, std)
        if episode > 0:
            print("At episode {}, {}".format(episode, desc))
        elif step > 0:
            print("At step {}, {}".format(step, desc))

    def plot(self, name, values, interval=10):
        indices = list(range(0, len(values), interval))
        means = []
        stds = []
        for i in indices:
            _values = values[i:(i + interval)]
            means.append(np.mean(_values))
            stds.append(np.std(_values))
        means = np.array(means)
        stds = np.array(stds)
        plt.figure()
        plt.title("{} History".format(name))
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds,
                         alpha=0.1, color="g")
        plt.plot(indices, means, "o-", color="g",
                 label="{} per {} episode".format(name.lower(), interval))
        plt.legend(loc="best")
        plt.show()

        # Save plot image to WandB
        wandb.log({f"{name}_plot": plt})

    def write(self, index, name, value):
        # WandB へのメトリクスの記録
        wandb.log({name: value}, step=index)

    def write_image(self, index, frames):
    # framesは複数の状態ベクトルを含む
    # 4つの次元を分けてプロット（位置、速度、ポールの角度、ポールの速度）

        frames = np.array(frames)  # framesがリストの場合、numpy配列に変換
        steps = np.arange(len(frames))  # フレームごとのステップ番号
        
        # plt.figure(figsize=(10, 6))
        
        # # 各状態ベクトルの要素をプロット
        # plt.subplot(4, 1, 1)
        # plt.plot(steps, frames[:, :, 0], label="Position")
        # plt.ylabel("Position")
        # plt.grid(True)
        
        # plt.subplot(4, 1, 2)
        # plt.plot(steps, frames[:, :, 1], label="Velocity", color="orange")
        # plt.ylabel("Velocity")
        # plt.grid(True)
        
        # plt.subplot(4, 1, 3)
        # plt.plot(steps, frames[:, :, 2], label="Pole Angle", color="green")
        # plt.ylabel("Pole Angle")
        # plt.grid(True)
        
        # plt.subplot(4, 1, 4)
        # plt.plot(steps, frames[:, :, 3], label="Pole Velocity", color="red")
        # plt.ylabel("Pole Velocity")
        # plt.xlabel("Step")
        # plt.grid(True)
        
        # plt.tight_layout()
        
        # # グラフを保存
        # image_path = f"cartpole_states_step_{index}.png"
        # plt.savefig(image_path)
        # plt.close()

        # WandBに画像としてアップロード
        # image = wandb.Image(image_path, caption=f"CartPole States up to step {index}")
        # wandb.log({f"State Graph {index}": image})