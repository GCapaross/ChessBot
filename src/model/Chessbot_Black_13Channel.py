import chess
import numpy as np
import pickle
import torch

# Base model
from models.architecture_2Conv_classes.model import old_ChessModel_13

# Utils
piece_to_idx = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

# Constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_WEIGHTS = './models/architecture_2Conv_classes/blackOnly_13/epoch-45.pth'
MOVE_DICTIONARY = "../dataset/processed/test_elite/move_dictionary_13.p"

class ChessBot_13:
    def __init__(self):
        with open(MOVE_DICTIONARY, "rb") as f:
            self.move_dictionary = pickle.load(f)
        self.reverse_move_dictionary = {v: k for k, v in self.move_dictionary.items()}
        self.trained_model = old_ChessModel_13(len(self.move_dictionary)).to(device)
        self.trained_model.load_state_dict(torch.load(MODEL_WEIGHTS))
        self.trained_model.eval()  # Set model to evaluation mode

    def get_best_moves(self, board: chess.Board): 
        # Convert board to bitmap
        bitmaps = self.__board_to_tensor(board)
        
        for valid_move in board.legal_moves:
            to_square_idx = valid_move.to_square
            # Set the corresponding index in bitmaps to 1
            bitmaps[12][to_square_idx // 8][to_square_idx % 8] = 1

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
        tensor = np.zeros((13, 8, 8), dtype=np.float32)
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