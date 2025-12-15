import os
import serial
import threading
import time
import csv
import datetime
import re
import socket
from pynput.keyboard import Key, Listener
from AppKit import NSWorkspace

# =============================
# 設定
# =============================
PRINTER_PORT = "/dev/cu.wchusbserial110"
PRINTER_BAUD = 115200

# ロードセル側のポートは実機に合わせて書き換えてください
LOADCELL_PORT = "/dev/tty.usbserial-D30DO6IO"
LOADCELL_BAUD = 9600
USE_LOADCELL = True  # ロードセルがまだ無ければ False に

LOG_INTERVAL_SEC = 0.01  # 100Hzでログ
UDP_ENABLED = True
UDP_ADDR = ("127.0.0.1", 5005)  # Processing側がlistenする先

MOVE_DISTANCE_X = 100
MOVE_DISTANCE_Y = 100
MOVE_DISTANCE_Z = 0.01          # y/Y用 0.01mm
MOVE_DISTANCE_Z_FINE = 0.001    # u/U用 0.001mm
MOVE_DISTANCE_Z_COARSE = 0.1    # t/T用 0.1mm
MOVE_VELOCITY = 300
FACTOR = 10

MIN_X, MAX_X = 15, 1000
MIN_Z, MAX_Z = -2.0, 45

# ★追加: 自動押し込みのパラメータ
AUTO_TARGET_KG = 0.05      # ここを変えれば目標荷重を変えられる
AUTO_STEP_Z = -0.001       # 1回でどれだけ押し込むか（マイナスで下方向）
AUTO_INTERVAL = 0.1     # 何秒ごとに押し込むか

# =============================
# グローバル状態
# =============================
ser_printer = serial.Serial(PRINTER_PORT, PRINTER_BAUD, timeout=1, write_timeout=1)
ser_loadcell = None
if USE_LOADCELL:
    ser_loadcell = serial.Serial(LOADCELL_PORT, LOADCELL_BAUD, timeout=1)

command_buffer = []
shift_pressed = False

state_lock = threading.Lock()
global_state = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "last_cmd": "",
    "last_printer": "",
    "load_raw": "",
    "load_value": None,
    "load_unit": "",
    # 追加したフィールド
    "trial_id": 0,
    "last_z_op": "",
    "last_move_interval": "",
    "last_move_freq": "",
}

# UDPソケット
udp_sock = None
if UDP_ENABLED:
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ===== ディレクトリの用意 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

CSV_FILENAME = os.path.join(
    LOG_DIR,
    f"printer_loadcell_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
)

# =============================
# ユーティリティ
# =============================
def update_position(axis, distance):
    with state_lock:
        if axis == "X":
            global_state["x"] += distance
        elif axis == "Y":
            global_state["y"] += distance
        else:
            global_state["z"] += distance
        x = global_state["x"]
        y = global_state["y"]
        z = global_state["z"]
    print(f"Pos → X:{x}, Y:{y}, Z:{z}")

def serial_send(gcode):
    command_buffer.append(gcode)
    with state_lock:
        global_state["last_cmd"] = gcode
    print(f"Queued: {gcode}")
    if len(command_buffer) == 1:
        send_next_command()

def send_next_command():
    ser_printer.write((command_buffer[0] + "\r\n").encode())
    print(f"Sent: {command_buffer[0]}")

def is_terminal_frontmost():
    front_app = NSWorkspace.sharedWorkspace().frontmostApplication().localizedName()
    print(front_app)
    return front_app in ["Terminal", "iTerm2", "iTerm"]

# =============================
# プリンタからの読み取りスレッド
# =============================
def serial_read_printer():
    while True:
        line = ser_printer.readline().decode(errors='ignore').strip()
        if line:
            print(f"Printer → {line}")
            with state_lock:
                global_state["last_printer"] = line
            if "ok" in line and command_buffer:
                command_buffer.pop(0)
                if command_buffer:
                    send_next_command()

# =============================
# ロードセルからの読み取りスレッド
# (OpenScale的な「数値＋単位」の1行が来る想定)
# =============================
loadcell_line_re = re.compile(r"([-+]?\d+\.?\d*)\s*([a-zA-Z]+)?")

def serial_read_loadcell():
    if ser_loadcell is None:
        return
    while True:
        line = ser_loadcell.readline().decode(errors="ignore").strip()
        if not line:
            continue

        # 例: "70400,0.0997,kg,23.2500,0,"
        parts = line.split(",")
        weight_val = None
        unit = ""
        if len(parts) >= 3:
            try:
                weight_val = float(parts[1])
            except ValueError:
                weight_val = None
            unit = parts[2]  # "kg"

        with state_lock:
            global_state["load_raw"] = line
            global_state["load_value"] = weight_val
            global_state["load_unit"] = unit

# =============================
# ログスレッド (CSV + UDP)
# =============================
def log_loop():
    with open(CSV_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iso_time", "epoch",
            "x", "y", "z",
            "load_value", "load_unit", "load_raw",
            "last_cmd", "last_printer",
            "trial_id", "last_z_op",
            "last_move_interval", "last_move_freq",
        ])
        while True:
            time.sleep(LOG_INTERVAL_SEC)
            now = time.time()
            iso = datetime.datetime.fromtimestamp(now).isoformat()
            with state_lock:
                row = [
                    iso, f"{now:.6f}",
                    global_state["x"],
                    global_state["y"],
                    global_state["z"],
                    global_state["load_value"] if global_state["load_value"] is not None else "",
                    global_state["load_unit"],
                    global_state["load_raw"],
                    global_state["last_cmd"],
                    global_state["last_printer"],
                    global_state.get("trial_id", ""),
                    global_state.get("last_z_op", ""),
                    global_state.get("last_move_interval", ""),
                    global_state.get("last_move_freq", ""),
                ]
            writer.writerow(row)
            f.flush()

            if UDP_ENABLED and udp_sock is not None:
                z_op = global_state.get("last_z_op", "")
                msg = f"{iso},{row[2]},{row[3]},{row[4]},{row[5]},{z_op}\n"
                udp_sock.sendto(msg.encode(), UDP_ADDR)

# =============================
# Zバースト実行
# =============================
def run_z_burst(step, count=10, interval=0.05):
    """
    Z軸を step mm ずつ、interval秒ごとにcount回動かす
    """
    # このバースト全体の設定を記録
    with state_lock:
        global_state["last_move_interval"] = interval
        global_state["last_move_freq"] = (1.0 / interval) if interval > 0 else ""

    for i in range(count):
        # 範囲チェック
        with state_lock:
            nz = global_state["z"] + step
        if nz < MIN_Z or nz > MAX_Z:
            print(f"Z制限(バースト): step={step}, i={i}")
            break

        update_position('Z', step)
        with state_lock:
            global_state["last_z_op"] = f"{step:+.4f}"
            global_state["trial_id"] = global_state.get("trial_id", 0) + 1
            # interval/freqはすでにセット済み

        vel = MOVE_VELOCITY
        serial_send(f"G1 Z{step} F{vel}")

        time.sleep(interval)

# =============================
# ★追加: ロードセルが一定値に達するまで押し込む
# =============================
def auto_press_to_load(target_kg=AUTO_TARGET_KG,
                       step=AUTO_STEP_Z,
                       interval=AUTO_INTERVAL,
                       max_steps=200):
    if not USE_LOADCELL:
        print("auto_press_to_load: ロードセルが有効ではありません")
        return

    # このモードの間隔を記録（CSVに残す用）
    with state_lock:
        global_state["last_move_interval"] = interval
        global_state["last_move_freq"] = (1.0 / interval) if interval > 0 else ""

    for i in range(max_steps):
        # 現在のロードセル値とZを読む
        with state_lock:
            load_val = global_state["load_value"]
            z_now = global_state["z"]

        # 目標に達したら終了
        if load_val is not None and load_val >= target_kg:
            print(f"auto_press_to_load: target reached ({load_val:.4f} kg)")
            break

        # Zの範囲チェック
        if z_now + step < MIN_Z or z_now + step > MAX_Z:
            print("auto_press_to_load: Z制限に達したので停止します")
            break

        # 実際に1ステップ押し込む
        update_position('Z', step)
        with state_lock:
            global_state["last_z_op"] = f"{step:+.4f}"
            global_state["trial_id"] = global_state.get("trial_id", 0) + 1
        serial_send(f"G1 Z{step} F{MOVE_VELOCITY}")

        time.sleep(interval)

# =============================
# キー入力処理
# =============================
def handle_xy_diagonal(key_char):
    dx = 0; dy = 0
    if key_char == 'q': dx, dy = -MOVE_DISTANCE_X,  MOVE_DISTANCE_Y
    if key_char == 'e': dx, dy =  MOVE_DISTANCE_X,  MOVE_DISTANCE_Y
    if key_char == 'z': dx, dy = -MOVE_DISTANCE_X, -MOVE_DISTANCE_Y
    if key_char == 'c': dx, dy =  MOVE_DISTANCE_X, -MOVE_DISTANCE_Y

    with state_lock:
        new_x = global_state["x"] + dx
    if new_x < MIN_X or new_x > MAX_X:
        print(f"X制限: Δ{dx}")
        dx = 0

    if dx: update_position('X', dx)
    if dy: update_position('Y', dy)

    vel = MOVE_VELOCITY * (FACTOR if shift_pressed else 1)
    cmd = "G1"
    if dx: cmd += f" X{dx}"
    if dy: cmd += f" Y{dy}"
    cmd += f" F{vel}"
    serial_send(cmd)

def on_press(key):
    global shift_pressed

    if key == Key.shift:
        shift_pressed = True

    # XY移動
    if hasattr(key, 'char') and key.char and key.char.lower() in ['w','a','s','d']:
        mv = MOVE_DISTANCE_Y if key.char.lower() in ['w','s'] else MOVE_DISTANCE_X
        axis = 'Y' if key.char.lower() in ['w','s'] else 'X'
        mv = -mv if key.char.lower() in ['s','a'] else mv
        vel = MOVE_VELOCITY * (FACTOR if shift_pressed else 1)

        if axis == 'X':
            with state_lock:
                nx = global_state["x"] + mv
            if nx < MIN_X or nx > MAX_X:
                print(f"{axis}制限: Δ{mv}")
                return

        update_position(axis, mv)
        serial_send(f"G1 {axis}{mv} F{vel}")

    # 対角
    elif hasattr(key, 'char') and key.char and key.char.lower() in ['q','e','z','c']:
        handle_xy_diagonal(key.char.lower())

    # Z軸 0.01mm (y/Y)
    elif hasattr(key, 'char') and key.char and key.char.lower() == 'y':
        dz = MOVE_DISTANCE_Z
        dz = -dz if key.char.islower() else dz
        vel = MOVE_VELOCITY * (FACTOR if shift_pressed else 1)
        with state_lock:
            nz = global_state["z"] + dz
        if nz < MIN_Z or nz > MAX_Z:
            print(f"Z制限: Δ{dz}")
            return
        update_position('Z', dz)
        with state_lock:
            global_state["last_z_op"] = f"{dz:+.4f}"
            global_state["trial_id"] = global_state.get("trial_id", 0) + 1
            global_state["last_move_interval"] = ""
            global_state["last_move_freq"] = ""
        serial_send(f"G1 Z{dz} F{vel}")

    # Z軸 0.001mm (u/U)
    elif hasattr(key, 'char') and key.char and key.char.lower() == 'u':
        dz = MOVE_DISTANCE_Z_FINE
        dz = -dz if key.char.islower() else dz
        vel = MOVE_VELOCITY * (FACTOR if shift_pressed else 1)
        with state_lock:
            nz = global_state["z"] + dz
        if nz < MIN_Z or nz > MAX_Z:
            print(f"Z制限(微): Δ{dz}")
            return
        update_position('Z', dz)
        with state_lock:
            global_state["last_z_op"] = f"{dz:+.4f}"
            global_state["trial_id"] = global_state.get("trial_id", 0) + 1
            global_state["last_move_interval"] = ""
            global_state["last_move_freq"] = ""
        serial_send(f"G1 Z{dz} F{vel}")

    # Z軸 0.1mm (t/T)
    elif hasattr(key, 'char') and key.char and key.char.lower() == 't':
        dz = MOVE_DISTANCE_Z_COARSE
        dz = -dz if key.char.islower() else dz
        vel = MOVE_VELOCITY * (FACTOR if shift_pressed else 1)
        with state_lock:
            nz = global_state["z"] + dz
        if nz < MIN_Z or nz > MAX_Z:
            print(f"Z制限(粗): Δ{dz}")
            return
        update_position('Z', dz)
        with state_lock:
            global_state["last_z_op"] = f"{dz:+.4f}"
            global_state["trial_id"] = global_state.get("trial_id", 0) + 1
            global_state["last_move_interval"] = ""
            global_state["last_move_freq"] = ""
        serial_send(f"G1 Z{dz} F{vel}")

    # 連続Zテスト: 0.01mmを一定間隔で10回
    elif hasattr(key, 'char') and key.char and key.char.lower() == 'b':
        threading.Thread(
            target=run_z_burst,
            args=(-0.01, 10, 0.05),
            daemon=True
        ).start()

    # 連続Zテスト: 0.001mmを一定間隔で10回
    elif hasattr(key, 'char') and key.char and key.char.lower() == 'n':
        threading.Thread(
            target=run_z_burst,
            args=(-0.001, 10, 0.05),
            daemon=True
        ).start()

    # ★追加: pでロードセル0.1kgまで押し込む
    elif hasattr(key, 'char') and key.char and key.char.lower() == 'p':
        threading.Thread(
            target=auto_press_to_load,
            daemon=True
        ).start()

    if key == Key.esc:
        return False

def on_release(key):
    global shift_pressed
    if key == Key.shift:
        shift_pressed = False

# =============================
# 初期化とメイン
# =============================
def wait_empty():
    while command_buffer:
        time.sleep(0.05)

if __name__ == "__main__":
    # プリンタ読み取り開始
    threading.Thread(target=serial_read_printer, daemon=True).start()

    # ロードセル読み取り開始
    if USE_LOADCELL:
        threading.Thread(target=serial_read_loadcell, daemon=True).start()

    # ログ開始
    threading.Thread(target=log_loop, daemon=True).start()

    # ホーミング＆相対モード
    for c in ["G28 X", "G28 Y", "G28 Z", "G91"]:
        serial_send(c)
        wait_empty()

    # 初期位置例
    for c,axis,dist in [("G1 Z45 F9000","Z",45),("G1 X15 F9000","X",15),("G1 Z-10 F9000","Z",-10)]:
        serial_send(c); wait_empty(); update_position(axis, dist)

    # キーボード待ち
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
