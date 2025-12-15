# sensor_reader.py
import time
import datetime
import re
import logging
import serial
import numpy as np
from multiprocessing import Process, Queue

def sensor_process(tactile_queue):

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
    
    tactile_ser = None
    try:
        tactile_ser = serial.Serial('/dev/cu.usbmodem101', 115200, timeout=1)
        logging.info("センサのシリアルポートが初期化されました。")

        touch_counter = 0  # 触覚センサの読み取りカウンターを初期化

        while True:
            now = datetime.datetime.now()
            centiseconds = int(now.microsecond / 10000)  # 1e6μs -> 1e2cs
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S.") + f"{centiseconds:02d}"
            tactile_data = read_touch_sensor(tactile_ser)
            if tactile_data:
                cantilever_1, cantilever_2, cantilever_3 = tactile_data
                state = np.array([cantilever_1, cantilever_2, cantilever_3], dtype=np.float32)
                # タイムスタンプ付きでキューに送信
                tactile_queue.put((timestamp, state))
                touch_counter += 1

            time.sleep(0.00125)  # 約800Hz
    except KeyboardInterrupt:
        logging.info("Sensor process interrupted.")
    except Exception as e:
        logging.error(f"Unexpected error in sensor_process: {e}")
    finally:
        tactile_ser.close()
        if tactile_ser is not None:
            tactile_ser.close()
        logging.info("Sensor serial ports closed.")