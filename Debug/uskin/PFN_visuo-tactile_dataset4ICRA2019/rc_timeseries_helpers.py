# rc_timeseries_helpers.py
import torch
import torch.nn.functional as F

"""
==========================================================
【略語対応表（コメント内で使用）】
- B : batch_size（バッチ内の系列数）
- T : time_steps（時系列長。washout/stride適用後の長さ）
- C : num_classes / output_dim（出力次元。分類ならクラス数）
- H : reservoir_dim（リザバーのノード数＝状態次元）
- ch: channel_count（チャンネル数。例：[B, ch, T, ...] の ch）
- N : num_samples（リッジ回帰に渡すサンプル数。通常 N = B*T）
==========================================================

【本ヘルパーの設計方針】
- モデルの出力は「時系列次元 T を保ったまま」扱う：例 logits は [B, T, C]
- loss 計算も原則 [B, T, ...] のまま行う（CEの場合は PyTorch の都合で [B, C, T] に transpose）
- リッジ回帰更新時のみ内部で 2D に reshape する（view(C,-1) など “軸入れ替えの勘違い” を避ける）
"""

def infer_device_from_model(model, fallback="cpu"):
    """model のパラメータから device を推定する（推定できない場合 fallback）"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)

def extract_states_time_major(esn_output):
    """
    ESN の出力を [batch_size, time_steps, reservoir_dim] (= [B, T, H]) に正規化する。

    想定入力：
    - [B, ch, T, H]
    - [B, T, H]
    """
    if esn_output.dim() == 4:
        # [B, ch, T, H] -> ch=0 を採用して [B, T, H]
        return esn_output[:, 0]
    if esn_output.dim() == 3:
        return esn_output
    raise ValueError(f"ESN出力の形状が想定外です: shape={tuple(esn_output.shape)}")

def extract_logits_time_major(readout_output):
    """
    ReadOut の出力を [batch_size, time_steps, output_dim] (= [B, T, C]) に正規化する。

    想定入力：
    - [B, ch, T, C]
    - [B, T, C]
    """
    if readout_output.dim() == 4:
        return readout_output[:, 0]
    if readout_output.dim() == 3:
        return readout_output
    raise ValueError(f"ReadOut出力の形状が想定外です: shape={tuple(readout_output.shape)}")

def apply_time_selection(x_time_major, washout_steps=0, time_stride=1):
    """
    時系列次元 T を保ったまま、washout と stride を適用する。
    x_time_major は先頭2次元が [B, T, ...] を想定。
    """
    washout_steps = max(int(washout_steps), 0)
    time_stride = max(int(time_stride), 1)
    x_sel = x_time_major[:, washout_steps::time_stride]
    if x_sel.shape[1] == 0:
        raise ValueError("washout/strideの結果、time_steps が 0 になりました。設定を見直してください。")
    return x_sel

def prepare_time_distributed_targets(labels, batch_size, time_steps, num_classes, device):
    """
    ラベルを以下の2つに変換する（時系列次元Tを保つ）：
    - targets_onehot_time_major : [B, T, C] float（MSE/BCEやリッジ回帰用）
    - target_index_per_sequence : [B] long（精度計算・CE用）

    対応する labels 形式：
    A) 系列ごとの one-hot : [B, C] float
    B) 系列ごとの index   : [B] long または [B,1] long
    C) 時刻ごとの index   : [B, T] long
    D) 時刻ごとの one-hot : [B, T, C] float
    """
    # D) [B, T, C] one-hot
    if labels.dim() == 3 and labels.shape == (batch_size, time_steps, num_classes):
        targets_onehot = labels.float().to(device)
        target_index = targets_onehot[:, 0].argmax(dim=-1).long()  # 代表として t=0 のラベル
        return targets_onehot, target_index

    # C) [B, T] index
    if labels.dim() == 2 and labels.shape == (batch_size, time_steps) and labels.dtype in (torch.int64, torch.int32):
        target_index_BT = labels.long().to(device)
        targets_onehot = F.one_hot(target_index_BT, num_classes=num_classes).float()  # [B, T, C]
        target_index = target_index_BT[:, 0]
        return targets_onehot, target_index

    # A) [B, C] one-hot
    if labels.dim() == 2 and labels.shape == (batch_size, num_classes) and labels.dtype.is_floating_point:
        labels_onehot = labels.float().to(device)
        target_index = labels_onehot.argmax(dim=-1).long()
        targets_onehot = labels_onehot.unsqueeze(1).expand(batch_size, time_steps, num_classes)  # [B, T, C]
        return targets_onehot, target_index

    # B) [B] または [B,1] index
    if labels.dim() == 1:
        target_index = labels.long().to(device)
    elif labels.dim() == 2 and labels.shape == (batch_size, 1):
        target_index = labels[:, 0].long().to(device)
    else:
        raise ValueError(f"labels形式が想定外です: shape={tuple(labels.shape)}, dtype={labels.dtype}")

    targets_onehot = F.one_hot(target_index, num_classes=num_classes).float().unsqueeze(1).expand(batch_size, time_steps, num_classes)
    return targets_onehot, target_index

def compute_loss_time_kept(logits_time_major, labels, criterion):
    """
    時系列次元 T を保ったまま loss を計算する。
    logits_time_major: [B, T, C]

    - CrossEntropyLoss の場合：PyTorch流儀に合わせて logits を [B, C, T] に変換し、
      target を [B, T] (class index) として渡す。
    - それ以外（MSELoss/BCEWithLogitsLossなど）：targets を [B, T, C] にしてそのまま渡す。

    戻り値：
    - loss（スカラー）
    - targets_onehot_time_major: [B, T, C]（リッジ更新にも使える）
    - target_index_per_sequence: [B]（精度計算用）
    """
    batch_size, time_steps, num_classes = logits_time_major.shape
    device = logits_time_major.device

    targets_onehot, target_index = prepare_time_distributed_targets(
        labels=labels,
        batch_size=batch_size,
        time_steps=time_steps,
        num_classes=num_classes,
        device=device
    )

    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        # CE は入力が [B, C, T]、ターゲットが [B, T] を受け取れる
        logits_BCT = logits_time_major.transpose(1, 2)  # [B, C, T]
        target_BT = target_index[:, None].expand(batch_size, time_steps)  # 系列ラベルを時刻方向に複製
        loss = criterion(logits_BCT, target_BT)
        return loss, targets_onehot, target_index

    # MSE / BCEWithLogitsLoss 等（[B, T, C] のまま）
    loss = criterion(logits_time_major, targets_onehot)
    return loss, targets_onehot, target_index

def sequence_accuracy_majority_vote(logits_time_major, target_index_per_sequence):
    """
    時系列全体で多数決（ヒストグラム投票）して系列ラベルを推定し、精度を返す。
    logits_time_major: [B, T, C]
    target_index_per_sequence: [B]
    """
    batch_size, time_steps, num_classes = logits_time_major.shape

    predicted_index_BT = logits_time_major.argmax(dim=-1)  # [B, T]
    vote_counts_BC = F.one_hot(predicted_index_BT, num_classes=num_classes).sum(dim=1).float()  # [B, C]
    predicted_index_B = vote_counts_BC.argmax(dim=-1)  # [B]
    # print(f"predicted_index_B + {predicted_index_B}")
    # print(f"target_index_per_sequence + {target_index_per_sequence}")

    return (predicted_index_B == target_index_per_sequence).float().mean().item()
