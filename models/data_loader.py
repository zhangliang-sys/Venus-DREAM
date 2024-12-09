import json
import torch
from torch.utils.data import Dataset

SEQUENCE_MAX_LENGTH = 1022

class ProteinpHDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.data = []
        for item in data:
            processed_item = {
                'opt_sequence': item['opt_sequence'][:SEQUENCE_MAX_LENGTH],
                'opt_pH': item['opt_pH'],
                'env_sequences': [seq[:SEQUENCE_MAX_LENGTH] for seq in item['env_sequences']],
                'env_pHs': item['env_pHs'],
                'opt_id': item['opt_id'],
                'env_ids': item['env_ids']
            }
            self.data.append(processed_item)
        print(f"Loaded {len(self.data)} items, all sequences truncated to max length {SEQUENCE_MAX_LENGTH}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'opt_seq': item['opt_sequence'],
            'opt_pH': torch.tensor(float(item['opt_pH'])).float(),
            'env_seqs': item['env_sequences'],
            'env_pHs': torch.tensor([float(pH) for pH in item['env_pHs']]).float(),
            'opt_id': item['opt_id'],
            'env_ids': item['env_ids']
        }

class SupportDataset(Dataset):
    def __init__(self, accs, sequences, pHs):
        self.accs = accs
        self.sequences = sequences
        self.pHs = pHs

    def __len__(self):
        return len(self.accs)

    def __getitem__(self, idx):
        return self.accs[idx], self.sequences[idx], self.pHs[idx]

    @staticmethod
    def collate_fn(batch):
        accs, sequences, pHs = zip(*batch)
        return list(accs), list(sequences), torch.tensor(pHs)




