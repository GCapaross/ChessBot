import json
import numpy as np
import polars as pl
import random
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch

VALIDATION_SIZE = 50_000

class ChessEvalDataset(Dataset):
    def __init__(self, file: str, load_batch_size = 6_400, validation_size = VALIDATION_SIZE, headers = ("bitmaps", "movePlayed", "validMoves")):
        self.lazy_dataset = pl.scan_csv(file, has_header=False, new_columns=headers)
        self.batch_size = load_batch_size
        self.feature_col = "bitmaps"
        self.target_col = "movePlayed"

        self.validation_size = validation_size
        self.total_rows = self.lazy_dataset.select(pl.len()).collect().item() - self.validation_size
        # self.total_rows = 2_500_000 

        self.cached_batches: dict[int, tuple] = {}
        self.cached_batch_id: int | None = None

    def __len__(self):
        return self.total_rows
        # return 2_500_000  # For testing purposes, we can limit the dataset size
    
    def get_validation_set(self):
        """Get the validation set"""
        # Fetch the last rows of the dataset
        validation_dataset = (self.lazy_dataset
                            .slice(self.total_rows, self.validation_size)
                            .collect())
        
        # Process features and target
        features = validation_dataset.select(self.feature_col)
        features = torch.tensor(np.array([self.convert_to_array_bitmap(bitmaps) for bitmaps in features["bitmaps"]]), dtype=torch.float32)
        
        played_moves = validation_dataset.select(self.target_col).to_numpy()
        targets = torch.tensor(played_moves, dtype=torch.long)

        return features, targets


    def _get_batch(self, batch_id):
        """Load a specific batch of data, wrapped with lru_cache for memory management"""
        # Calculate batch range
        if batch_id == self.cached_batch_id:
            return self.cached_batches[batch_id]
        
        self.cached_batches.pop(self.cached_batch_id, None) # Delete old batch
        
        # Calculate batch range
        start_idx = batch_id * self.batch_size
        end_idx = min((batch_id + 1) * self.batch_size, self.total_rows)
        self.idxs = [i for i in range(end_idx - start_idx)]
        random.shuffle(self.idxs) # Shuffle the indices for the batch
        
        # Fetch only this batch of data using offset and limit
        batch_dataset = (self.lazy_dataset
                    .slice(start_idx, end_idx - start_idx)
                    .collect())
        
        # Process features and target
        features = batch_dataset.select(self.feature_col)
        features = torch.tensor(np.array([self.convert_to_array_bitmap(bitmaps) for bitmaps in features["bitmaps"]]), dtype=torch.float32)
        
        played_moves = batch_dataset.select(self.target_col).to_numpy()
        targets = torch.tensor(played_moves, dtype=torch.long)

        self.cached_batch_id = batch_id
        self.cached_batches[self.cached_batch_id] = (features, targets)
        
        return self.cached_batches[batch_id]

    def __getitem__(self, idx):
        # Calculate which batch this index belongs to
        batch_id = idx // self.batch_size

        # Get the batch
        features, targets = self._get_batch(batch_id)

        # Get the item from the batch
        idx_in_batch = self.idxs[idx % self.batch_size]  # Get the next index from the shuffled list sorted(range(0, len(batch))) [ ]

        return features[idx_in_batch], targets[idx_in_batch]
    


    def convert_to_array_bitmap(self, row: str):
        """"
        Converts board into array

        :param str row: Board with side to play;
                            This must represent a np.ndarray[np.int64, shape=(13)] 
        :return: np.ndarray[shape(13, 8, 8), dtype=np.float32]]:
        """
        NUM_CHANNELS = 12
        boards = np.array(json.loads(row), dtype=np.uint64) # shape = (13,)
        array = np.empty((NUM_CHANNELS, 8, 8), dtype=np.uint8)

        # Set pieces boards
        board_int8_view = boards.view(dtype=np.uint8).reshape((NUM_CHANNELS, 8)) # shape = (13, 8)
        board_as_int = np.unpackbits(board_int8_view, axis=1, bitorder='little').reshape((NUM_CHANNELS, 8, 8))
        array[:] = board_as_int

        # Set side to play board
        # array[12] = np.ones(shape = (1, 8, 8)) if boards[12] == 1 else np.zeros(shape=(1,8,8))
        return array
    





class ChessEvalDataset_13(Dataset):
    def __init__(self, file: str, load_batch_size = 6_400, validation_size = VALIDATION_SIZE, headers = ("bitmaps", "movePlayed", "validMoves")):
        self.lazy_dataset = pl.scan_csv(file, has_header=False, new_columns=headers)
        self.batch_size = load_batch_size
        self.feature_col = "bitmaps"
        self.target_col = "movePlayed"

        self.validation_size = validation_size
        self.total_rows = self.lazy_dataset.select(pl.len()).collect().item() - self.validation_size
        # self.total_rows = 2_500_000 

        self.cached_batches: dict[int, tuple] = {}
        self.cached_batch_id: int | None = None

    def __len__(self):
        # return self.total_rows
        return 2_500_000  # For testing purposes, we can limit the dataset size
    
    def get_validation_set(self):
        """Get the validation set"""
        # Fetch the last rows of the dataset
        validation_dataset = (self.lazy_dataset
                            .slice(self.total_rows, self.validation_size)
                            .collect())
        
        # Process features and target
        features = validation_dataset.select(self.feature_col)
        features = torch.tensor(np.array([self.convert_to_array_bitmap(bitmaps) for bitmaps in features["bitmaps"]]), dtype=torch.float32)
        
        played_moves = validation_dataset.select(self.target_col).to_numpy()
        targets = torch.tensor(played_moves, dtype=torch.long)

        return features, targets


    def _get_batch(self, batch_id):
        """Load a specific batch of data, wrapped with lru_cache for memory management"""
        # Calculate batch range
        if batch_id == self.cached_batch_id:
            return self.cached_batches[batch_id]
        
        self.cached_batches.pop(self.cached_batch_id, None) # Delete old batch
        
        # Calculate batch range
        start_idx = batch_id * self.batch_size
        end_idx = min((batch_id + 1) * self.batch_size, self.total_rows)
        self.idxs = [i for i in range(end_idx - start_idx)]
        random.shuffle(self.idxs) # Shuffle the indices for the batch
        
        # Fetch only this batch of data using offset and limit
        batch_dataset = (self.lazy_dataset
                    .slice(start_idx, end_idx - start_idx)
                    .collect())
        
        # Process features and target
        features = batch_dataset.select(self.feature_col)
        features = torch.tensor(np.array([self.convert_to_array_bitmap(bitmaps) for bitmaps in features["bitmaps"]]), dtype=torch.float32)
        
        played_moves = batch_dataset.select(self.target_col).to_numpy()
        targets = torch.tensor(played_moves, dtype=torch.long)

        self.cached_batch_id = batch_id
        self.cached_batches[self.cached_batch_id] = (features, targets)
        
        return self.cached_batches[batch_id]

    def __getitem__(self, idx):
        # Calculate which batch this index belongs to
        batch_id = idx // self.batch_size

        # Get the batch
        features, targets = self._get_batch(batch_id)

        # Get the item from the batch
        idx_in_batch = self.idxs[idx % self.batch_size]  # Get the next index from the shuffled list sorted(range(0, len(batch))) [ ]

        return features[idx_in_batch], targets[idx_in_batch]
    


    def convert_to_array_bitmap(self, row: str):
        """"
        Converts board into array

        :param str row: Board with side to play;
                            This must represent a np.ndarray[np.int64, shape=(13)] 
        :return: np.ndarray[shape(13, 8, 8), dtype=np.float32]]:
        """
        NUM_CHANNELS = 13
        boards = np.array(json.loads(row), dtype=np.uint64) # shape = (13,)
        array = np.empty((NUM_CHANNELS, 8, 8), dtype=np.uint8)

        # Set pieces boards
        board_int8_view = boards.view(dtype=np.uint8).reshape((NUM_CHANNELS, 8)) # shape = (13, 8)
        board_as_int = np.unpackbits(board_int8_view, axis=1, bitorder='little').reshape((NUM_CHANNELS, 8, 8))
        array[:] = board_as_int

        return array