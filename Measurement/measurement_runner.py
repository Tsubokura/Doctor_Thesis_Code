import argparse
import csv
import os
import time
import json
import datetime
from multiprocessing import Queue, Process
from sensor_reader import sensor_process
from printer_manager import PrinterSerialManager

def run_measurements(distance_x: float,
                     distance_y: float,
                     speed: float,
                     count: int,
                     z_lift: float,
                     base_dir: str,
                     material: str,
                     environment_params: dict,
                     printer_port: str = "/dev/cu.wchusbserial1120"):

    # 1) 素材フォルダを作成
    mat_dir = os.path.join(base_dir, material)
    os.makedirs(mat_dir, exist_ok=True)

    # 2) 今回の測定バッチ用フォルダ名を「YYYYMMDD_HHMMSS」で作成
    batch_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(mat_dir, batch_ts)
    os.makedirs(batch_dir, exist_ok=True)

    # 3) バッチ単位の metadata.json
    batch_meta = {
        "material": material,
        "distance_x": distance_x,
        "distance_y": distance_y,
        "speed": speed,
        "z_lift": z_lift,
        "count": count,
        "batch_timestamp": batch_ts
    }
    with open(os.path.join(batch_dir, "batch_metadata.json"), "w") as f:
        json.dump(batch_meta, f, indent=2)

    # 4) マスターCSV（バッチが増えるごとに追記）
    master_csv = os.path.join(mat_dir, "index.csv")
    header = ["batch_ts", "run_no", "direction", "speed", "z_lift", "file_path"]
    if not os.path.exists(master_csv):
        with open(master_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # 5) センサープロセス起動
    tactile_queue = Queue()
    sensor_proc = Process(target=sensor_process, args=(tactile_queue,))
    sensor_proc.start()

    # 6) プリンタ接続
    log_path = os.path.join(batch_dir, "printer_log.csv")
    printer = PrinterSerialManager(environment_params, port=printer_port, log_path=log_path)

    # (A) 原点復帰
    printer.send_command("G28 X Y Z")
    printer.send_command("G91")
    printer.wait_idle(3.0)
    printer.send_command("G1 Y60 F2000")
    printer.wait_idle(3.0)

    # 7) 測定ループ
    for i in range(1, count + 1):
        # 各測定のファイル名パラメータ文字列
        direction = f"dx{distance_x}_dy{distance_y}"
        fname = f"run{i:02d}_{direction}_spd{speed}_lift{z_lift}.csv"
        csv_path = os.path.join(batch_dir, fname)

        print(f"\n--- Measurement {i}/{count} → {fname} ---")

        

        # (B) XY往復
        printer.send_command(f"G1 X{distance_x} Y{distance_y} F{int(speed)}")
        printer.wait_idle()
        printer.send_command(f"G1 X{-distance_x} Y{-distance_y} F{int(speed)}")
        printer.wait_idle()

        # (C) Z方向リフト往復
        printer.send_command(f"G1 Z{z_lift} F{int(speed)}")
        printer.wait_idle()
        printer.send_command(f"G1 Z{-z_lift} F{int(speed)}")
        printer.wait_idle()

        # (D) データ収集
        data = []
        while not tactile_queue.empty():
            ts, arr = tactile_queue.get_nowait()
            data.append([ts, *arr.tolist()])

        if data:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "s1", "s2", "s3"])
                writer.writerows(data)
            print(f"Saved → {csv_path}")

            # マスターCSVに追記
            with open(master_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([batch_ts, i, direction, speed, z_lift, csv_path])
        else:
            print("Warning: No data collected for this run.")

    # 8) クリーンアップ
    printer.close()
    sensor_proc.terminate()
    sensor_proc.join()
    print("=== All done ===")

if __name__ == "__main__":
    measurement_params = {
        "distance_x": 50,
        "distance_y": 50,
        "speed": 1000,
        "count": 3,
        "z_lift": 5,
        "base_dir": "./test",
        "material": "debug_hogehoge",
    }
    environment_params = {"MIN_X": -100, "MAX_X": 100, "MIN_Z": 0, "MAX_Z": 50}

    run_measurements(
        distance_x=measurement_params["distance_x"],
        distance_y=measurement_params["distance_y"],
        speed=measurement_params["speed"],
        count=int(measurement_params["count"]),
        z_lift=measurement_params["z_lift"],
        base_dir=measurement_params["base_dir"],
        material=measurement_params["material"],
        environment_params=environment_params,
        printer_port="/dev/cu.wchusbserial1120"
    )
