import serial
import time

#3Dプリンターへの移動命令が実行できること自体を確認するスクリプト

ser = serial.Serial("/dev/cu.wchusbserial120", 115200, timeout=1)
time.sleep(2)  # ボーレート変更後など安定のため待つ

# ホームコマンドを送る
ser.write(b"G28\r\n")
time.sleep(0.1)

while True:
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if line:
        print(f"Read: {line}")
        if 'ok' in line.lower():
            print("Got OK")
            break

