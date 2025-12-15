#!/usr/bin/env python3
"""
Z軸座標ごとのロードセル値の散らばりを調べるスクリプト（設定はコード内で指定）

・CSV は、少なくとも以下の列を持つことを想定:
    - z
    - load_value

・MODE を "single" にすると、TARGET_Z 付近のデータだけを抽出して統計量を表示
・MODE を "all" にすると、全ての Z ごとに load_value の統計量を一覧表示
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ================== 設定ここから ==================

# 入力 CSV ファイル
CSV_PATH = Path("./logs/printer_loadcell_20251118_111738.csv")   # ここを書き換えてください

# 動作モード: "single" なら特定の Z, "all" なら全 Z ごとに集計
MODE = "single"              # "single" または "all"

# MODE == "single" のときに使う設定
TARGET_Z = 39.6              # 調べたい Z の値
TOL = 1e-6                   # Z の許容誤差 (TARGET_Z ± TOL を同一とみなす)

# MODE == "all" のときに使う設定
MIN_COUNT = 1                # 各 Z ごとに最低限必要なデータ数

# ================== 設定ここまで ==================


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    CSV/TSV を読み込む。
    区切り文字は自動推定 (カンマでもタブでもOK)。
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")
    return df


def to_numeric(series: pd.Series) -> pd.Series:
    """数値列に変換（空文字などは NaN に落とす）"""
    return pd.to_numeric(series, errors="coerce")


def describe_one_z(df: pd.DataFrame, target_z: float, tol: float = 1e-6) -> None:
    """特定の Z について load_value の散らばりを表示する"""
    if "z" not in df.columns or "load_value" not in df.columns:
        raise ValueError("必要な列 'z' または 'load_value' が CSV にありません。")

    df = df.copy()
    df["z"] = to_numeric(df["z"])
    df["load_value"] = to_numeric(df["load_value"])

    # Z が target_z ± tol の行を抽出
    mask = df["z"].notna() & df["load_value"].notna() & (np.abs(df["z"] - target_z) <= tol)
    sub = df.loc[mask, "load_value"]

    print(f"=== Z ≈ {target_z} (tol = {tol}) のロードセル値の散らばり ===")
    print(f"データ点数: {len(sub)}")

    if len(sub) == 0:
        print("該当するデータがありません。")
        return

    mean = sub.mean()
    var = sub.var(ddof=1)  # 不偏分散
    std = sub.std(ddof=1)
    vmin = sub.min()
    vmax = sub.max()
    q25, q50, q75 = sub.quantile([0.25, 0.5, 0.75])

    print(f"平均値       : {mean:.6g}")
    print(f"分散         : {var:.6g}")
    print(f"標準偏差     : {std:.6g}")
    print(f"最小値       : {vmin:.6g}")
    print(f"最大値       : {vmax:.6g}")
    print(f"25%点 (Q1)   : {q25:.6g}")
    print(f"50%点 (中央値): {q50:.6g}")
    print(f"75%点 (Q3)   : {q75:.6g}")


def describe_all_z(df: pd.DataFrame, min_count: int = 1) -> None:
    """全ての Z について load_value の散らばりを一覧表示"""
    if "z" not in df.columns or "load_value" not in df.columns:
        raise ValueError("必要な列 'z' または 'load_value' が CSV にありません。")

    df = df.copy()
    df["z"] = to_numeric(df["z"])
    df["load_value"] = to_numeric(df["load_value"])

    valid = df[df["z"].notna() & df["load_value"].notna()]

    # Z ごとに統計量を計算
    grouped = (
        valid
        .groupby("z")["load_value"]
        .agg(
            count="count",
            mean="mean",
            var=lambda s: s.var(ddof=1) if len(s) > 1 else np.nan,
            std=lambda s: s.std(ddof=1) if len(s) > 1 else np.nan,
            min="min",
            max="max",
        )
        .reset_index()
    )

    # 指定より少ないデータ点しかない Z は除外
    grouped = grouped[grouped["count"] >= min_count]
    grouped = grouped.sort_values("z")

    if grouped.empty:
        print("有効な Z ごとの統計量がありません。")
        return

    # 表形式で表示
    pd.set_option("display.max_rows", None)
    print("=== Z ごとのロードセル値の散らばり ===")
    print(f"※ count < {min_count} の Z は除外しています")
    print(
        grouped.to_string(
            index=False,
            formatters={
                "z":   lambda v: f"{v:.6g}",
                "mean": lambda v: f"{v:.6g}",
                "var": lambda v: f"{v:.6g}" if pd.notna(v) else "NaN",
                "std": lambda v: f"{v:.6g}" if pd.notna(v) else "NaN",
                "min": lambda v: f"{v:.6g}",
                "max": lambda v: f"{v:.6g}",
            },
        )
    )


def main():
    df = load_dataframe(CSV_PATH)

    if MODE == "single":
        describe_one_z(df, TARGET_Z, TOL)
    elif MODE == "all":
        describe_all_z(df, MIN_COUNT)
    else:
        raise ValueError(f"未知の MODE が指定されています: {MODE}")


if __name__ == "__main__":
    main()
