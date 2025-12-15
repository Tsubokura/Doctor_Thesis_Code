#!/usr/bin/env python3
"""
ある Z 軸目標値 TARGET_Z の近傍に到達した「試行」ごとに、
ロードセル値 (load_value) の散らばり（分散など）を出すスクリプト。

前提となる列:
    - z
    - load_value
    - iso_time (任意: 区間の開始/終了時刻を出すのに利用)
    - trial_id (任意: あれば trial ごとに集計)

試行(アプローチ)の定義:
    - abs(z - TARGET_Z) <= Z_TOL を満たすサンプルが連続している区間を 1 試行とする。
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ================== 設定ここから ==================

# 入力 CSV ファイル
CSV_PATH = Path("./logs/printer_loadcell_20251118_174558.csv")   # ←ここを実際のファイル名に変えてください

# 調べたい Z の値と許容誤差
TARGET_Z = 38.205   # 例: 0.0 mm
Z_TOL    = 0.0001      # 例: ±0.01 mm の範囲を「到達」とみなす

# 1 試行として認めるための最小サンプル数
MIN_SAMPLES_PER_TRIAL = 3

# trial を区別する列名 (無ければ None にする)
TRIAL_ID_COLUMN = "trial_id"  # CSV に無ければ None にしてください

# ================== 設定ここまで ==================


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    CSV/TSV を読み込む。
    区切り文字は自動推定 (カンマ/タブなど)。
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    return df


def to_numeric(series: pd.Series) -> pd.Series:
    """数値列に変換（空文字などは NaN に落とす）"""
    return pd.to_numeric(series, errors="coerce")


def find_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    真理値配列 mask (True のところが Z 近傍) から、
    True が連続している区間の (start_idx, end_idx) を列挙。
    """
    idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
        return []

    segments: list[tuple[int, int]] = []
    start = idxs[0]
    prev = idxs[0]

    for i in idxs[1:]:
        if i == prev + 1:
            # 連続中
            prev = i
        else:
            # ここまでの区間を確定
            segments.append((start, prev))
            start = i
            prev = i
    # 最後の区間を追加
    segments.append((start, prev))
    return segments


def analyze_one_trial(df_trial: pd.DataFrame,
                      trial_label) -> list[dict]:
    """
    1 つの trial（trial_id が同じグループ）について、
    TARGET_Z 近傍に到達した区間ごとに load_value の統計量を計算。
    """
    # index を 0..N-1 に振り直して扱いやすくする
    df_t = df_trial.reset_index(drop=True).copy()

    # 数値化
    df_t["z"] = to_numeric(df_t["z"])
    df_t["load_value"] = to_numeric(df_t["load_value"])

    # Z, load_value の有効値のみ対象
    valid_mask = df_t["z"].notna() & df_t["load_value"].notna()

    # 目標 Z 近傍 (abs(z - TARGET_Z) <= Z_TOL)
    near_mask = valid_mask & (np.abs(df_t["z"] - TARGET_Z) <= Z_TOL)

    segments = find_segments(near_mask.to_numpy())

    results = []

    for seg_idx, (start_i, end_i) in enumerate(segments, start=1):
        sub = df_t.iloc[start_i:end_i + 1]

        if len(sub) < MIN_SAMPLES_PER_TRIAL:
            # サンプル数が少なすぎる区間は無視（必要ならコメントアウト）
            continue

        load = sub["load_value"]
        z_vals = sub["z"]

        mean = load.mean()
        var = load.var(ddof=1) if len(load) > 1 else np.nan
        std = load.std(ddof=1) if len(load) > 1 else np.nan
        vmin = load.min()
        vmax = load.max()

        z_mean = z_vals.mean()
        z_min = z_vals.min()
        z_max = z_vals.max()

        t_start = sub["iso_time"].iloc[0] if "iso_time" in sub.columns else None
        t_end   = sub["iso_time"].iloc[-1] if "iso_time" in sub.columns else None

        results.append(
            {
                "trial_id": trial_label,
                "approach_idx": seg_idx,
                "n_samples": len(sub),
                "z_mean": z_mean,
                "z_min": z_min,
                "z_max": z_max,
                "load_mean": mean,
                "load_var": var,
                "load_std": std,
                "load_min": vmin,
                "load_max": vmax,
                "t_start": t_start,
                "t_end": t_end,
            }
        )

    return results


def main():
    df = load_dataframe(CSV_PATH)

    if "z" not in df.columns or "load_value" not in df.columns:
        raise ValueError("CSV に 'z' 列または 'load_value' 列がありません。")

    all_results: list[dict] = []

    # trial 分けの有無を確認
    if TRIAL_ID_COLUMN is not None and TRIAL_ID_COLUMN in df.columns:
        # trial_id ごとに処理
        for trial_label, df_trial in df.groupby(TRIAL_ID_COLUMN):
            res = analyze_one_trial(df_trial, trial_label)
            all_results.extend(res)
    else:
        # trial_id が無い場合は，ファイル全体を 1 trial とみなす
        res = analyze_one_trial(df, trial_label="all")
        all_results.extend(res)

    if not all_results:
        print("TARGET_Z 近傍に到達した試行が見つかりませんでした。")
        return

    summary_df = pd.DataFrame(all_results)

    # Z に近い順に並べる（ほぼ同じはずだが念のため）
    summary_df = summary_df.sort_values(["trial_id", "approach_idx"])

    pd.set_option("display.max_rows", None)
    print(f"=== Z ≈ {TARGET_Z} (±{Z_TOL}) に到達した試行ごとのロードセル統計量 ===")
    print(summary_df.to_string(index=False,
                               formatters={
                                   "z_mean":     lambda v: f"{v:.6g}",
                                   "z_min":      lambda v: f"{v:.6g}",
                                   "z_max":      lambda v: f"{v:.6g}",
                                   "load_mean":  lambda v: f"{v:.6g}",
                                   "load_var":   lambda v: f"{v:.6g}" if pd.notna(v) else "NaN",
                                   "load_std":   lambda v: f"{v:.6g}" if pd.notna(v) else "NaN",
                                   "load_min":   lambda v: f"{v:.6g}",
                                   "load_max":   lambda v: f"{v:.6g}",
                               }))

    # おまけ: 全試行の load_mean / load_var の概要をざっくり出す
    print("\n--- 試行ごとの load_mean の概要 ---")
    print(summary_df["load_mean"].describe())

    print("\n--- 試行ごとの load_var の概要 ---")
    print(summary_df["load_var"].describe())


if __name__ == "__main__":
    main()
