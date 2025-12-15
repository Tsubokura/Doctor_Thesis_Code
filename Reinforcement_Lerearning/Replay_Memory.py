import random

# class RecurrentExperienceReplayMemory:
#     def __init__(self, capacity, sequence_length=10):
#         self.capacity = capacity
#         self.memory = []
#         self.seq_length = sequence_length

#     def push(self, transition):
#         self.memory.append(transition)
#         if len(self.memory) > self.capacity:
#             del self.memory[0]

#     def sample(self, batch_size):
#         if len(self.memory) < self.seq_length:
#             return []
#         finish = random.sample(range(self.seq_length, len(self.memory)), batch_size)
#         sequences = []
#         for end in finish:
#             start = end - self.seq_length
#             sequence = self.memory[start:end]
#             # エピソードの境界をまたがないようにチェック
#             if any(exp.d for exp in sequence[:-1]):
#                 # エピソード境界をまたぐシーケンスはスキップ
#                 continue
#             sequences.append(sequence)
#         if len(sequences) < batch_size and len(sequences) > 0:
#             # シーケンスが足りない場合、既存のものを使って補完
#             sequences += random.choices(sequences, k=batch_size - len(sequences))
#         return sequences

#     def __len__(self):
#         return len(self.memory)

class RecurrentExperienceReplayMemory:
    def __init__(self, capacity, sequence_length=10):
        self.capacity = capacity
        self.memory = []
        self.seq_length = sequence_length

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        """
        最新のデータから sequence_length×batch_size 個ぶんの経験を連続で取り出し、
        それを seq_length ごとに分割して返す。
        """
        required_length = self.seq_length * batch_size
        if len(self.memory) < required_length:
            # 最新データだけで十分なシーケンス数を確保できない場合
            return []

        # メモリ末尾から「batch_size×sequence_length」個分を取り出す
        end = len(self.memory)
        start = end - required_length
        block = self.memory[start:end]

        sequences = []
        # block を seq_length 単位に分割する
        for i in range(batch_size):
            seq_start = i * self.seq_length
            seq_end = seq_start + self.seq_length
            sequence = block[seq_start:seq_end]

            # エピソードの境界(途中で done=True)をまたいでいないかチェック
            if any(exp.d for exp in sequence[:-1]):
                # エピソード途中に done があればスキップ
                continue

            sequences.append(sequence)

        # スキップが多くて sequences が足りない場合は、同じものを重複サンプリングして補完
        if len(sequences) < batch_size and len(sequences) > 0:
            sequences += random.choices(sequences, k=batch_size - len(sequences))

        return sequences

    def __len__(self):
        return len(self.memory)
