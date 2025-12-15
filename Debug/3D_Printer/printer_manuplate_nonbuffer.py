import serial
import threading
import time
from pynput.keyboard import Key, Listener

# シリアルポート設定
ser = serial.Serial("/dev/cu.wchusbserial110", 115200, timeout=1, write_timeout=1)
# ser = serial.Serial("/dev/cu.wchusbserial120", 115200, timeout=1, write_timeout=1)

MOVE_DISTANCE_X = 100
MOVE_DISTANCE_Y = 100
MOVE_DISTANCE_Z = 0.01          # y/Y用 0.01mm
MOVE_DISTANCE_Z_FINE = 0.001    # u/U用 0.001mm
MOVE_DISTANCE_Z_COARSE = 0.1    # t/T用 0.1mm ←追加
MOVE_VELOCITY = 300
FACTOR = 10

# 移動制限の閾値
MIN_X, MAX_X = 15, 1000
MIN_Z, MAX_Z = -2.0, 45

command_buffer = []
shift_pressed = False

current_position = {"X": 0, "Y": 0, "Z": 0}

def update_position(axis, distance):
    current_position[axis] += distance
    print(f"Pos → X:{current_position['X']}, Y:{current_position['Y']}, Z:{current_position['Z']}")

def serial_send(gcode):
    command_buffer.append(gcode)
    print(f"Queued: {gcode}")
    if len(command_buffer) == 1:
        send_next_command()

def send_next_command():
    ser.write((command_buffer[0] + "\r\n").encode())
    print(f"Sent: {command_buffer[0]}")

def handle_xy_diagonal(key_char):
    # q: ↖, e: ↗, z: ↙, c: ↘
    dx = 0; dy = 0
    if key_char == 'q': dx, dy = -MOVE_DISTANCE_X,  MOVE_DISTANCE_Y
    if key_char == 'e': dx, dy =  MOVE_DISTANCE_X,  MOVE_DISTANCE_Y
    if key_char == 'z': dx, dy = -MOVE_DISTANCE_X, -MOVE_DISTANCE_Y
    if key_char == 'c': dx, dy =  MOVE_DISTANCE_X, -MOVE_DISTANCE_Y

    new_x = current_position['X'] + dx
    if new_x < MIN_X or new_x > MAX_X:
        print(f"X制限: 現{current_position['X']} Δ{dx}")
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

    if hasattr(key, 'char') and key.char and key.char.lower() in ['w','a','s','d']:
        mv = MOVE_DISTANCE_Y if key.char.lower() in ['w','s'] else MOVE_DISTANCE_X
        axis = 'Y' if key.char.lower() in ['w','s'] else 'X'
        mv = -mv if key.char.lower() in ['s','a'] else mv
        vel = MOVE_VELOCITY * (FACTOR if shift_pressed else 1)

        if axis == 'X':
            nx = current_position['X'] + mv
            if nx < MIN_X or nx > MAX_X:
                print(f"{axis}制限: 現{current_position[axis]} Δ{mv}")
                return

        update_position(axis, mv)
        serial_send(f"G1 {axis}{mv} F{vel}")

    elif hasattr(key, 'char') and key.char and key.char.lower() in ['q','e','z','c']:
        handle_xy_diagonal(key.char.lower())

    # Z軸 0.01mm (y/Y)
    elif hasattr(key, 'char') and key.char and key.char.lower() == 'y':
        dz = MOVE_DISTANCE_Z
        dz = -dz if key.char.islower() else dz  # yでマイナス, Yでプラス
        vel = MOVE_VELOCITY * (FACTOR if shift_pressed else 1)
        nz = current_position['Z'] + dz
        if nz < MIN_Z or nz > MAX_Z:
            print(f"Z制限: 現{current_position['Z']} Δ{dz}")
            return
        update_position('Z', dz)
        serial_send(f"G1 Z{dz} F{vel}")

    # Z軸 0.001mm (u/U)
    elif hasattr(key, 'char') and key.char and key.char.lower() == 'u':
        dz = MOVE_DISTANCE_Z_FINE
        dz = -dz if key.char.islower() else dz  # uでマイナス, Uでプラス
        vel = MOVE_VELOCITY * (FACTOR if shift_pressed else 1)
        nz = current_position['Z'] + dz
        if nz < MIN_Z or nz > MAX_Z:
            print(f"Z制限(微): 現{current_position['Z']} Δ{dz}")
            return
        update_position('Z', dz)
        serial_send(f"G1 Z{dz} F{vel}")

    # Z軸 0.1mm (t/T) ←ここが今回追加したところ
    elif hasattr(key, 'char') and key.char and key.char.lower() == 't':
        dz = MOVE_DISTANCE_Z_COARSE
        dz = -dz if key.char.islower() else dz  # tでマイナス, Tでプラス
        vel = MOVE_VELOCITY * (FACTOR if shift_pressed else 1)
        nz = current_position['Z'] + dz
        if nz < MIN_Z or nz > MAX_Z:
            print(f"Z制限(粗): 現{current_position['Z']} Δ{dz}")
            return
        update_position('Z', dz)
        serial_send(f"G1 Z{dz} F{vel}")

    if key == Key.esc:
        return False

def on_release(key):
    global shift_pressed
    if key == Key.shift:
        shift_pressed = False

def serial_read():
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"Printer → {line}")
            if "ok" in line and command_buffer:
                command_buffer.pop(0)
                if command_buffer:
                    send_next_command()

def start_reading():
    threading.Thread(target=serial_read, daemon=True).start()

def wait_empty():
    while command_buffer:
        time.sleep(0.1)

if __name__ == "__main__":
    start_reading()

    # ホーミング＆初期化
    for c in ["G28 X", "G28 Y", "G28 Z", "G91"]:
        serial_send(c); wait_empty()

    # 初期位置例
    for c,axis,dist in [("G1 Z45 F9000","Z",45),("G1 X15 F9000","X",15),("G1 Z-0 F9000","Z",-0)]:
        serial_send(c); wait_empty(); update_position(axis, dist)

    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
