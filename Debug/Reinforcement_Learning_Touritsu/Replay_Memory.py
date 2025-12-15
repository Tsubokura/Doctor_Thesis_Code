import random

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
        if len(self.memory) < self.seq_length:
            return []
        finish = random.sample(range(self.seq_length, len(self.memory)), batch_size)
        sequences = []
        for end in finish:
            start = end - self.seq_length
            sequence = self.memory[start:end]
            # エピソードの境界をまたがないようにチェック
            if any(exp.d for exp in sequence[:-1]):
                # エピソード境界をまたぐシーケンスはスキップ
                continue
            sequences.append(sequence)
        if len(sequences) < batch_size and len(sequences) > 0:
            # シーケンスが足りない場合、既存のものを使って補完
            sequences += random.choices(sequences, k=batch_size - len(sequences))
        return sequences

    def __len__(self):
        return len(self.memory)