import serial
import time
import logging

#シリアル通信を通じ，測距センサと触覚センサの計測自体を行えること，そして10秒あたりにおけるそれぞれのセンサの計測量を確認するコード

# ログの設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#両方のセンサからデータが送られてきたら受信処理をするため，測距センサの計測時間に計測量が依存する．
def test_sensor_serial(duration=10):
    try:
        # シリアルポートの初期化
        sensor_ser = serial.Serial('/dev/cu.usbmodem1101', 115200, timeout=1)
        distance_ser = serial.Serial('/dev/cu.usbmodem2101', 115200, timeout=1)
        logging.info("Sensor serial ports initialized for testing.")
        
        start_time = time.time()  # 開始時刻の記録
        sample_count = 0  # サンプル数のカウント初期化

        while time.time() - start_time < duration:
            # 両方のシリアルポートにデータがある場合
            if sensor_ser.in_waiting > 0 and distance_ser.in_waiting > 0:
                # センサーからのデータ読み取り
                sensor_data = sensor_ser.readline().decode('utf-8', errors='ignore').strip()
                # 距離センサーからのデータ読み取り
                distance_data = distance_ser.readline().decode('utf-8', errors='ignore').strip()
                
                # デバッグログにデータを出力
                # logging.debug(f"Sensor Data: {sensor_data}")
                # logging.debug(f"Distance Data: {distance_data}")
                
                sample_count += 1  # サンプル数をカウントアップ
            
            time.sleep(0.000125)  # 短い休止

        # 10秒後に取得したサンプル数をログに記録
        logging.info(f"Total samples collected in {duration} seconds: {sample_count}")
    
    except KeyboardInterrupt:
        logging.info("Test interrupted by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        # シリアルポートを閉じる
        sensor_ser.close()
        distance_ser.close()
        logging.info("Sensor serial ports closed.")

if __name__ == "__main__":
    test_sensor_serial()
