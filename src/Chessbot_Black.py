import chess
import numpy as np
import pickle
import torch

# Base model
from model.architecture_2Conv_classes.model import ChessModel as ChessModel

# Utils
piece_to_idx = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

# Constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_WEIGHTS = './models/architecture_2Conv_classes/blackOnly_batchnorm/epoch-80.pth'
MODEL_WEIGHTS = './model/architecture_2Conv_classes/blackOnly_7kernel/epoch-35.pth'
MOVE_DICTIONARY = "./model/move_dictionary.p"

class ChessBot:
    def __init__(self):
        with open(MOVE_DICTIONARY, "rb") as f:
            self.move_dictionary = pickle.load(f)
        self.reverse_move_dictionary = {v: k for k, v in self.move_dictionary.items()}
        self.trained_model = ChessModel(len(self.move_dictionary), kernel_size=7).to(device)
        self.trained_model.load_state_dict(torch.load(MODEL_WEIGHTS))
        self.trained_model.eval()  # Set model to evaluation mode

    def get_best_moves(self, board: chess.Board): 
        # Convert board to bitmap
        bitmaps = self.__board_to_tensor(board)
        
        # Model predicts best move
        probs = self.__predict_move(bitmaps)

        # Recebmos as classes dos moves: converter para moves e depois ordernar os moves por probabilidade
        moves = [self.reverse_move_dictionary[idx] for idx, prob in enumerate(probs)]
        best_moves = sorted(
            zip(moves, probs),
            key=lambda x: x[1], # Sort by probability
            reverse=True
        )
        return best_moves
    
    def __board_to_tensor(self, board: chess.Board):
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                tensor[piece_to_idx[piece.symbol()]][square // 8][square % 8] = 1
        return tensor

    def __predict_move(self, bitmaps):
        bitmaps  = np.expand_dims(bitmaps, axis=0)
        X_tensor = torch.tensor(bitmaps, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits = self.trained_model(X_tensor)
        
        logits = logits.squeeze(0)  # Remove batch dimension
        
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
        
        return probabilities