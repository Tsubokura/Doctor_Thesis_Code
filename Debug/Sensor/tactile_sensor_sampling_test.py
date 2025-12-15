#シリアル通信を通じ，触覚センサの計測自体を行えること，そして10秒あたりの触覚センサの計測量を確認するコード

import serial
import time

# シリアルポートの設定（環境に合わせて変更）
# 修正: 'ensor_ser' のタイポを修正
ser = serial.Serial('/dev/cu.usbmodem1101', 115200, timeout=1)

time.sleep(2)  # シリアル通信が安定するまで待つ

print("Start collecting data for 10 seconds...")

try:
    start_time = time.time()  # 開始時刻の記録
    duration = 10  # 計測時間（秒）
    data_count = 0  # データカウントの初期化

    while time.time() - start_time < duration:
        if ser.in_waiting > 0:  # 受信バッファにデータがある場合
            data = ser.readline().decode('utf-8', errors='ignore').strip()  # 受信データを読み込み、デコード
            if data:  # 空文字列でない場合にカウント
                # print(f"Value: {data} value")  # 距離データを表示
                data_count += 1  # カウントをインクリメント

        # 非同期的にデータをチェックするために短い待機を入れる
        time.sleep(0.000125)  # 1msの待機

    
    print(f"\nTotal data received in {duration} seconds: {data_count} samples")

except KeyboardInterrupt:
    print("\nProgram terminated by user.")
finally:
    ser.close()  # 終了時にシリアルポートを閉じる
    print("Serial port closed.")

