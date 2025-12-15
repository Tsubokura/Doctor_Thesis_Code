
import threading
import multiprocessing
import time
import re
import gymnasium as gym
import numpy as np
import serial
import queue
import logging

from multiprocessing import Process, Queue
from fn_framework_torch import Observer

# ロックオブジェクトを作成
serial_lock = threading.Lock()

# イベントオブジェクトを作成
gcode_event = threading.Event()

class PrinterSerialManager:
    """
    シリアルポートへの送受信を一本化し、OKを受信してから次コマンドを送る。
    """

    def __init__(self, environment_params, port="/dev/cu.wchusbserial120", baudrate=115200, timeout=1):
        self.current_position = {"X": 0, "Z": 0}
        self.MIN_X = float(environment_params["MIN_X"])  # 例: X軸の最小位置
        self.MAX_X = float(environment_params["MAX_X"])  # 例: X軸の最大位置（必要なら）
        self.MIN_Z = float(environment_params["MIN_Z"])   # 例: Z軸の最小位置
        self.MAX_Z = float(environment_params["MAX_Z"])   # 例: Z軸の最大位置（必要なら）

        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # オープン直後に待機すると安定することが多い

        self.command_queue = queue.Queue()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        # "ok"を受信したらセットされるイベント
        self.ok_event = threading.Event()

        # 受信バッファ（行単位で処理しやすいように）
        self.receive_buffer = ""

        # スレッドの起動
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.read_thread.start()
        self.write_thread.start()

    def _read_loop(self):
        """
        シリアルポートからのレスポンスを受信し、"ok" を検出する。
        """
        while not self.stop_event.is_set():
            try:
                chunk = self.ser.read(128)  # 128バイトまとめて読む
                if chunk:
                    data = chunk.decode('utf-8', errors='ignore')
                    self.receive_buffer += data

                    # 改行区切りでパース
                    while '\n' in self.receive_buffer:
                        line, self.receive_buffer = self.receive_buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            print(f"[PrinterSerialManager] Recv: {line}")

                            if "busy" in line.lower():
                            # 現在のコマンドを確認
                                if self.current_command is not None:
                                    (cur_gcode, cur_id) = self.current_command
                                    print(f"[PrinterSerialManager] >>> 'busy' detected. Command ID={cur_id}, gcode={cur_gcode}")
                                else:
                                    print(f"[PrinterSerialManager] >>> 'busy' detected, but no current_command set.")

                            # "ok" という文字列が含まれたら OK イベントをセット
                            if "ok" in line.lower():
                                self.ok_event.set()
            except serial.SerialException as e:
                print(f"[PrinterSerialManager] SerialException in read loop: {e}")
                break
            except Exception as e:
                print(f"[PrinterSerialManager] Unexpected error in read loop: {e}")
                break

    def _write_loop(self):
        """
        command_queue から G-code を取り出し、送信して OK を待ってから次へ進む。
        """
        while not self.stop_event.is_set():
            try:
                # print(self.command_queue.get(timeout=0.1))
                gcode, cmd_id = self.command_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self.lock:
                # 新しいコマンドを送る前に OKイベントをクリア
                self.ok_event.clear()
                self.current_command = (gcode, cmd_id)
                # 改行を付与して送信
                self.ser.write((gcode + "\r\n").encode('utf-8'))
                print(f"[PrinterSerialManager] Sent: {gcode}")

            # OKが返るまで待機。タイムアウトは適宜調整。
            # タイムアウトしても次のコマンドを送るかどうかは要検討。
            ok_received = self.ok_event.wait(timeout=5)
            if not ok_received:
                print(f"[PrinterSerialManager] WARNING: No 'ok' received for '{gcode}' within timeout.")
                # ここで再送や中断などの処理を入れてもよい

    def send_command(self, gcode: str):
        """
        外部から G-code を送るときに呼ぶメソッド。
        """
        command_id = time.time()
        axis, value = self.extract_g1_movements(gcode)

        if axis == None and value == None:
            self.command_queue.put((gcode, command_id))
            return
        else:
            new_position = self.current_position[axis] + value

        # print(f"Movement: {{'{axis}': {value}}}")
        print(self.current_position)

        if axis == 'X':
            if new_position < self.MIN_X:
                print(f"X軸の移動が制限されました。現在の位置: {self.current_position['X']}, 移動予定量: {value}")
                return
            elif 'MAX_X' in globals() and new_position > self.MAX_X:
                print(f"X軸の移動が制限されました。現在の位置: {self.current_position['X']}, 移動予定量: {value}")
                return
            else:
                self.current_position[axis] += value
        elif axis == 'Z':
            if new_position < self.MIN_Z:
                print(f"Z軸の移動が制限されました。現在の位置: {self.current_position['Z']}, 移動予定量: {value}")
                return
            elif 'MAX_Z' in globals() and new_position > self.MAX_Z:
                print(f"Z軸の移動が制限されました。現在の位置: {self.current_position['Z']}, 移動予定量: {value}")
                return
            else:
                self.current_position[axis] += value

        self.command_queue.put((gcode, command_id))

    def update_position(self, axis, distance):
        self.current_position[axis] += distance

    def extract_g1_movements(self, gcode):
        g1_pattern = re.compile(r"^G1\s+([A-Z][-\d.\s]*)")
        axis_pattern = re.compile(r"([XYZ][-\d.]+)")


        g1_match = g1_pattern.search(gcode)
        if g1_match:
            movement = g1_match.group(1)
            axis = movement[0]  # The first character, e.g., 'X', 'Y', or 'Z'
            value = float(movement[1:])  # The remaining characters as a float value
            
            return axis, value
        return None, None

    def close(self):
        """
        スレッドを終了し、シリアルポートをクローズ。
        """
        self.stop_event.set()
        self.read_thread.join()
        self.write_thread.join()
        self.ser.close()
        print("[PrinterSerialManager] Closed.")

class TurningUpControlEnv(gym.Env):
    """
    カスタム環境: ロボットのセンサデータを観測として使用
    アクション: G-code コマンドの選択
    報酬: 4次元目のセンサデータに変換を施した値
    """
    def __init__(self, state_queue, reward_queue, actions, environment_params):
        super(TurningUpControlEnv, self).__init__()

        
        self.init_Z = int(environment_params["init_Z"])
        self.move_X = int(environment_params["move_X"])
        self.move_Velocity = int(environment_params["move_Velocity"])
        self.next_move_Velocity = int(environment_params["move_Velocity"])
        self.move_Velocity_initial = int(environment_params["move_Velocity"])

        #報酬を変換する際の定数
        self.inv_distance = float(environment_params["inv_distance"])
        self.threshold_distance = int(environment_params["threshold_distance"])

        self.state_length = int(environment_params["state_length"])
        self.reward_length = int(environment_params["reward_length"])

        self.reward_addition = float(environment_params["reward_addition"])
        self.reward_penalty = float(environment_params["reward_penalty"])

        self.state_queue = state_queue
        self.reward_queue = reward_queue
        self.actions = actions

        self.iftest = False

        self.action_space = gym.spaces.Discrete(len(actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(150,3), dtype=np.float32)
        self.current_state = np.zeros(3, dtype=np.float32)
        
        self.move_complete = threading.Event()

        # シリアル送信用バッファ
        self.command_buffer_init = []

        # 各軸の移動距離の合計を保持
        self.total_distance = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}

        self.observer = PaparTurnUPObserver(self)

        try:
            self.printer_manager = PrinterSerialManager(
                environment_params,
                port="/dev/cu.wchusbserial120", 
                baudrate=115200,
                timeout=1
            )
        except serial.SerialException as e:
            print(f"Could not open serial port: {e}")
            raise


    def reset(self):
        # 環境を初期状態にリセット
        self.current_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 初期状態を設定
        self.total_distance = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}  # 移動距離をリセット

        try:
            while True:
                self.state_queue.get_nowait()  # 全要素を取り出す
        except queue.Empty:
                pass  # キューが空になったら終了

        try:
            while True:
                self.reward_queue.get_nowait()  # 全要素を取り出す
        except queue.Empty:
                pass  # キューが空になったら終了

        print("Queue cleared")

        return self.current_state

    def sleep_at_currentpositon(self, seconds):
        #secondsミリ秒間待機．
        gcode_sleep = f"G4 P{seconds}"
        self.send_gcode_threaded(gcode_sleep)

    def downsample_and_fix_length(self, sensor_data, downsample_factor=10, target_length=150):
        """
        sensor_data: (N, D) または (N,) 次元の NumPy配列 (Nは時系列長、Dは特徴数)
        downsample_factor: ダウンサンプリングの間引き率 (10 ならば 1/10)
        target_length: 出力する最終的な時系列長

        戻り値: (target_length, D) または (target_length,) 次元の NumPy配列
        """
        # 1. ダウンサンプリング (step=downsample_factorで間引き)
        # downsampled = sensor_data[::downsample_factor]
        # print(downsampled.shape)

        N = len(sensor_data)
        
        if sensor_data.ndim == 1:
            D = None
        else:
            D = sensor_data.shape[1]
        
        # データ長が downsample_factor で割り切れるようにトリム
        trimmed_length = (N // downsample_factor) * downsample_factor
        trimmed_data = sensor_data[:trimmed_length]
        
        if D is None:
            # 1次元データの場合
            reshaped = trimmed_data.reshape(-1, downsample_factor)
            downsampled = reshaped.mean(axis=1)
        else:
            # 2次元データの場合
            reshaped = trimmed_data.reshape(-1, downsample_factor, D)
            downsampled = reshaped.mean(axis=1)
        
        print(f"ダウンサンプリング後の形状: {downsampled.shape}")

        # 2. 必要に応じて長さを固定
        current_length = len(downsampled)

        if current_length == 0:
            # センサデータがそもそも空の場合は、ゼロ配列で返す
            if sensor_data.ndim == 1:
                return np.zeros(target_length, dtype=sensor_data.dtype)
            else:
                return np.zeros((target_length, sensor_data.shape[1]), dtype=sensor_data.dtype)

        if current_length < target_length:
            # 足りない分を最後の値で埋める
            last_value = downsampled[-1]
            needed = target_length - current_length
            if sensor_data.ndim == 1:
                pad = np.full(needed, last_value, dtype=sensor_data.dtype)
            else:
                pad = np.tile(last_value, (needed, 1))
            downsampled = np.concatenate([downsampled, pad], axis=0)
        elif current_length > target_length:
            # target_length を超える部分を切り捨て
            downsampled = downsampled[:target_length]

        print(f"最終形状: {downsampled.shape}")

        # この時点で downsampled.shape == (target_length, D) または (target_length,)
        return downsampled

    def transform_distance_with_offset(self, distance_array: np.ndarray):
        """
        (30,) の時系列データが入ってきたとして、
        1. 最初の値をオフセットとして取り
        2. 各値からオフセットを引く
        3. その結果に対し逆数 (1 / (x + 1e-3)) を計算
        4. 最後に平均を取って単一のスカラ報酬として返す

        ※ 必要に応じて合計にするなど処理を変えても OK
        """
        if distance_array.size == 0:
            # 空配列の場合は適当なデフォルト値を返す
            return -0.5

        offset = distance_array[0]
        # オフセットを引く
        offset_data = distance_array - offset

        offset_data = -np.where(np.abs(offset_data) < self.threshold_distance, 0, offset_data)
        # 逆数を計算
        # reward_value = 1.0 / (offset_data + self.inv_distance)
        reward_value = offset_data 

        return reward_value

    def step_episode_equalstep(self, action):
        print("action : ", action)
        current_Z_position = self.printer_manager.current_position["Z"]

        #actionの値により，Z軸の初期位置を調整．
        
        if action == 0 or action ==1 or action ==4:
            self.printer_manager.send_command(self.actions[action])
        else:
            self.move_Velocity = self.move_Velocity + int(self.actions[action])
            if self.move_Velocity <= 800:
                self.move_Velocity = 800
        

        time.sleep(0.1)#入れてみる


        gcode = f"G1 X{self.move_X} F{self.move_Velocity}"
        self.printer_manager.send_command(gcode)
        
        done = False
        reward = 0.0

        # センサーデータと報酬の収集
        try:
            states = []
            rewards = []
            while not self.state_queue.empty():
                state = self.state_queue.get_nowait()
                states.append(state)

            # print(self.reward_queue.get_nowait())
            while not self.reward_queue.empty():
                reward = self.reward_queue.get_nowait()
                rewards.append(reward)

            # print(states.shape)
            # print(states)
            
            # states = np.array(states)←もしかしたら，この処理においてずれが発生するかも

            # states = self.observer.transform(states)
            # states = self.downsample_and_fix_length(states, 10, self.state_length)
            # print(states.shape)
            
            # rewards = np.array(rewards)

            # rewards = self.transform_distance_with_offset(rewards)
            # rewards = self.downsample_and_fix_length(rewards, 1, self.reward_length)

            # if action == 1:
            #     rewards += self.reward_addition #ヘッダが下方向へ行くように事前知識を導入．
            #     if current_Z_position == self.printer_manager.MIN_Z:
            #         rewards += self.reward_penalty #ヘッダが下限の時，下方向へ行くようならばペナルティを加える．
            
            # print(rewards)
            # print(rewards.shape)

            #状態と報酬のサイズを同じにする
            # if rewards.shape[0] != 0:
            #     rewards = np.full(states.shape[0], self.transform_distance_to_reward(rewards))

            # if states.size > 0:
            #     self.current_state = states[-1]  # 最新の状態で更新
            # else:
            #     # 新しい状態がない場合の処理
            #     states = np.array([self.current_state])

            # if rewards.size == 0:
            #     reward = -0.5  # 報酬が取得できなかった場合のデフォルト値
            #     done = True

        except multiprocessing.queues.Empty:
            # センサーデータが取得できなかった場合
            reward = -0.5
            done = True

        info = {}

        return states, rewards, done, info


    def render(self, mode='human'):
        # 必要に応じて実装
        pass

    def move_to_first_position(self):
        gcode_tofirstposition = f"G1 X{-self.move_X} F{self.move_Velocity_initial*3}"
        self.printer_manager.send_command(gcode_tofirstposition)

    def move_to_initial_position(self):
        #episode開始時の場所へ戻る
        current_Z_position = self.printer_manager.current_position["Z"]
        difference = self.init_Z  - current_Z_position #初期のZ軸位置: Z = 2 との差分を取る．

        gcode_to_initialposition = f"G1 Z{difference} F{self.move_Velocity*3}"
        self.printer_manager.send_command(gcode_to_initialposition)
        #速度もリセット
        self.move_Velocity = self.move_Velocity_initial
        self.next_move_Velocity = self.move_Velocity_initial

    def sleep_at_currentpositon(self, seconds):
        gcode_sleep = f"G4 P{seconds}"
        self.printer_manager.send_command(gcode_sleep)

    def first_move(self):
        #学習における，一番最初のヘッダ移動
        gcode_firstturning = f"G1 X{self.move_X} F{self.move_Velocity}"
        self.printer_manager.send_command(gcode_firstturning)
        self.move_to_first_position()

    def wait_for_init_buffer_empty(self, interval=0.01):
        time.sleep(interval)

    def init_procedure(self):
        """
        ここで環境初期化時に実行したい G-code 群をまとめて送る。
        その都度 wait_for_init_buffer_empty() を呼ぶと逐次同期的に実行される。
        """

        init_gcodes = [
            "G28 X", "G28 Z", "G91",
            "G1 Z8 F9000", "G1 X20 F9000", "G1 Z2 F9000"
        ]

        for gcode in init_gcodes:
            self.printer_manager.send_command(gcode)
            print(f"[Env] Queued G-code: {gcode}")

        print("[Env] Initialization procedure completed.")


    def printer_serial_close(self):
        """
        旧: self.printer_ser.close() ではなく manager.close() を呼ぶ。
        """
        self.printer_manager.close()
        print("Printer serial manager closed.")


class PaparTurnUPObserver(Observer):
    def __init__(self, env):
        super(PaparTurnUPObserver, self).__init__(env)


    @staticmethod
    def normalize_rows(array):
        """
        各特徴（列）ごとに最小値と最大値を計算し、-1から1の範囲にスケーリングします。
        
        Parameters:
            array (np.ndarray): 正規化対象の2次元配列。形状は (サンプル数, 特徴数)。
        
        Returns:
            np.ndarray: 正規化された配列。
        """
        # 各列の最小値と最大値を計算
        min_vals = array.min(axis=0, keepdims=True)
        max_vals = array.max(axis=0, keepdims=True)
        
        # -1から1の範囲にスケーリング
        normalized_array = 2 * (array - min_vals) / (max_vals - min_vals + 1e-8) - 1
        
        return normalized_array

    def transform(self, state):
        state = np.array(state)
        state = self.normalize_rows(state)  # クラスメソッドとして呼び出し
        return state

def sensor_process(state_queue, reward_queue):
    import logging
    import time
    import serial
    import re
    import numpy as np

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def read_touch_sensor(sensor_ser):
        try:
            sensor_data = sensor_ser.readline().decode('utf-8', errors='ignore').strip()
            sensor_matches = re.findall(r'-?\d+\.\d+', sensor_data)
            if sensor_matches:
                cantilever_1, cantilever_2, cantilever_3 = map(float, sensor_matches)
                return cantilever_1, cantilever_2, cantilever_3
            else:
                logging.warning(f"無効な触覚センサデータ: {sensor_data}")
        except Exception as e:
            logging.error(f"触覚センサデータの読み取りエラー: {e}")
        return None

    def read_distance_sensor(distance_ser):
        try:
            distance_data = distance_ser.readline().decode('utf-8', errors='ignore').strip()
            distance = float(distance_data) if distance_data else None
            if distance is not None:
                return distance
            else:
                logging.warning(f"無効な測距センサデータ: {distance_data}")
        except Exception as e:
            logging.error(f"測距センサデータの読み取りエラー: {e}")
        return None

    try:
        touch_ser = serial.Serial('/dev/cu.usbmodem1101', 115200, timeout=1)
        distance_ser = serial.Serial('/dev/cu.usbmodem2101', 115200, timeout=1)
        logging.info("センサのシリアルポートが初期化されました。")

        touch_counter = 0  # 触覚センサの読み取りカウンターを初期化

        while True:
            current_time = time.time()  # タイムスタンプ取得
            touch_data = read_touch_sensor(touch_ser)
            if touch_data:
                cantilever_1, cantilever_2, cantilever_3 = touch_data
                state = np.array([cantilever_1, cantilever_2, cantilever_3], dtype=np.float32)
                # タイムスタンプ付きでキューに送信
                state_queue.put((current_time, state))
                touch_counter += 1

                if touch_counter >= 50:
                    current_time_reward = time.time()  # 報酬データのタイムスタンプ
                    distance = read_distance_sensor(distance_ser)
                    if distance is not None:
                        reward_queue.put((current_time_reward, distance))
                    else:
                        reward_queue.put((current_time_reward, -0.5))
                    touch_counter = 0

            time.sleep(0.00125)  # 約800Hz
    except KeyboardInterrupt:
        logging.info("Sensor process interrupted.")
    except Exception as e:
        logging.error(f"Unexpected error in sensor_process: {e}")
    finally:
        touch_ser.close()
        distance_ser.close()
        logging.info("Sensor serial ports closed.")
