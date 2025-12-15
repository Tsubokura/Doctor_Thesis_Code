# printer_manager.py
import re
import threading
import datetime
import queue
import serial
import time
class PrinterSerialManager:
    """
    シリアルポートへの送受信を一本化し、OKを受信してから次コマンドを送る。
    """

    def __init__(self, environment_params, port="/dev/cu.wchusbserial1120", baudrate=115200, timeout=1, log_path=None):
         # X, Y, Z 全軸の現在位置を管理
        self.current_position = {"X": 0, "Y": 0, "Z": 0}
        # Y 軸の最小・最大が環境に含まれていなければデフォルトを設定
        self.MIN_X = float(environment_params.get("MIN_X", -1e6))
        self.MAX_X = float(environment_params.get("MAX_X",  1e6))
        self.MIN_Y = float(environment_params.get("MIN_Y", -1e6))
        self.MAX_Y = float(environment_params.get("MAX_Y",  1e6))
        self.MIN_Z = float(environment_params.get("MIN_Z", 0))
        self.MAX_Z = float(environment_params.get("MAX_Z", 50))

        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # オープン直後に待機すると安定することが多い

        self.command_queue = queue.Queue()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        # "ok"を受信したらセットされるイベント
        self.ok_event = threading.Event()

        # 受信バッファ（行単位で処理しやすいように）
        self.receive_buffer = ""

        self.log_file = open(log_path, "a") if log_path else None

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
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            print(f"[PrinterSerialManager][{timestamp}] Recv: {line}")

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

                                if self.log_file:
                                    recv_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                                    self.log_file.write(f"{recv_ts},RECV,{line}\n")
                                    self.log_file.flush()
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

                # 送信時刻（ミリ秒まで）
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                # 改行を付与して送信
                self.ser.write((gcode + "\r\n").encode('utf-8'))
                print(f"[PrinterSerialManager][{timestamp}] Sent: {gcode}")

                if self.log_file:
                    self.log_file.write(f"{timestamp},SENT,{gcode}\n")
                    self.log_file.flush()

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

        elif axis == 'Y':
            # Y軸制限チェック
            if new_position < self.MIN_Y or new_position > self.MAX_Y:
                print(f"Y軸の移動が制限されました。現在の位置: {self.current_position['Y']}, 移動予定量: {value}")
                return
            self.current_position['Y'] = new_position
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
        if self.log_file:
            self.log_file.close()
        print("[PrinterSerialManager] Closed.")

    def wait_idle(self, timeout: float = None):
        """
        OK を待つか、指定秒数だけ待つ。
        引数 timeout を指定すると time.sleep()、未指定なら OK イベントを待機します。
        """
        if timeout is not None:
            time.sleep(timeout)
        else:
            # 次のコマンド送信まで OK が返ってくるのを待つ
            self.ok_event.wait()
