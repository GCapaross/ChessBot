# ChessBot


What style and approach can we take:
- Predict the best move
- Evaluate board positions
- Plays like a human (imitate a playing style)
- Play to win (as strong as possible)



Have in consideration book moves (Opening, rules, exceptions)

FEN strings
PGN strings
Tensor
Bitboard / Tensor representation
(Choose one of these)
(Transform into a bit map)


Options for model architecture:
- CNN (AlphaZero)
- Transformer-based models (Leela Chess)
- Reinforcement learning (Deeper things)

Frameworks:
- TensorFlow
- PyTorch

TO TRAIN:
- Calculate valid positions
- Mask illegal moves as negatives
- Augment data by flipping boards
After training:
- Predict move probabilities
- Mask illegal moves
- Highest probability move / Sample from probabilities (Human like move)


Python-chess for integration


Post execution:
- Reinforcement learning (Self play or MCTS)
- Model comprehension
- Evaluate ELO


Tools:
python-chess
pgnparser or chess.pgn
PyTorch or TensorFlow
numpy, pandas


Prioritize higher time controls and higher elos





Neural network with MinMaxing and Alpha-beta pruning


If we want we can ignore the dataset

[LINKS]

- https://healeycodes.com/building-my-own-chess-engine
- https://www.chessprogramming.org/Main_Page
- https://python-chess.readthedocs.io/en/latest/
- https://nanochess.org/chess3.html
- https://www.chess.com/games
- https://www.pgnmentor.com/files.html#openings

