# Chess Neural Net Engine
A simple PyTorch-based chess evaluator.

## How it Works
### Part 1: Tensorizing Data (`Game-Processor.py`)
1. Reads game data from a .pgn.zst file via a piece-by-piece decompression stream, so that the massive archive can stay compressed.
2. A loop begins through the archive until the requested number of games is met. Games that do not meet the specified criteria are skipped over.
3. If a given game meets the specified criteria, **every single position** in the game is stored to `tensor_list` after being tensorized to **(17,8,8)** as follows:
    - 17 "layers" of 8x8 (chessboard) grids, each layer representing another game aspect. Always stored from the **POV of the current player that must move in the position**.
    - Layers 0-5 are each type of the player-to-move's pieces, layers 6-11 for the opposition. 1 where a specific piece type is located, 0 elsewhere.
    - Layers 12-15 are the kingside and queenside castling rights of the to-move and opposing player respectively. 1 everywhere if legal, 0 everywhere if not.
    - Layer 16 locates any en passant squares.
    - The board is always vertically flipped to have the player-to-move side at the bottom. *Note: this means that when it is black to move, the king and queen side would appear flipped to us. This is so that the sides are always the same regardless of which side must move, making it simpler for the model to understand*.
    - Stored via `torch` as uint8 for space efficiency.
4. The outcome of the game is stored to `outcome_list` as an integer between -1 and 1. 1 represents the player-to-move winning, 0 a draw, and -1 a loss.
5. The move played from every given position is assigned a number code based on the number label of the square its coming from, and moving to. The labels are always oriented so that the promotion end is on squares 0-7. For non-promotions it is simply `(from) x 64 + (to) = (move code)`. For promotions, it follows `4096 + (to + 1) x (from + 1)` and then `+0` for queens, `+128` for rooks, `+256` for bishops, and `+384` for knights.
6. Once the requested number of games has been met, the positions and outcomes and moves are saved to a .pt file as uint8 and float32 and long respectively.
### Part 2: Deep Learning (`Model-Trainer.py`)
1. A simple dual headed model architecture is defined:
    - The backbone is 4x 2D convolutional layers defined by 32 in & out features (other than `cv1`, which must first take in the 17). They use 3x3 filters, and analyze the geometric dynamics of the position.
    - This is then split off into an evaluation head which scores which side is most likely to win, and a policy head which scores which moves are most likely to occur.
    - The value head is 2x simple linear layers, which filter it all down into a single evaluation number. This value is converted to a value between -1 and 1 via `tanh` and used in the final output.
    - The policy head is also 2x simple linear layers, which score all the move codes for most likely to occur in that position. Note that the output range 0-4608 is set by the highest possible move code, 4608. Only a small portion of this range are ever possible in a position.
2. The data is organized into batches of 64 [positions], 90% for training, 10% for validation. The training batches are randomly shuffled in order to ensure a consistent training supply. The validation batches are not shuffled since they bear no bearing over the model's learning, they are only for spectating its progress against unseen positions.
3. All of the prepared data is used in the training loop:
    - The model evaluates all the positions, and outputs its scoring in two tensors shape (64,1) and (64,4609).
    - The value loss is calculated by criterion `MSELoss`, and the policy loss by `CrossEntropyLoss`.
    - The model weights are adjusted by optimizer `Adam`.
4. Once the specified number of epochs are complete, the model is saved to a .pt file.
### Part 3: Basic Application (`Search-Algorithm.py` & `main.py`)
1. It is a applied into PUCT MCTS in `Search-Algorithm.py`, following the search formula `t/n + 2*grade* ( math.sqrt( nP+1 ) / (n + 1) )`, where t = total evaluation score, n = number of iterations, nP = number of parent node iterations, and grade is the model's assigned probability for the node (move) (0-1). It is called on using the function `Get_MCTS()` and applied in `main.py`.

## Steps to Replicate & Play the Model
1. Clone the repository.
2. Install the `chess` library, `zstandard`, and PyTorch.
3. Download a .pgn.zst compressed game archive from https://database.lichess.org/ (it might take a while to download), and place it into the folder.
4. Specify the archive file name, and your desired settings in `Game-Processor.py`, via the variables featured at the top.
5. Run `Game-Processor.py` (beware of RAM constraints).
6. Specify the tensor file name, and your desired settings in `Model-Trainer.py`, via the variables featured at the top.
7. Run `Model-Trainer.py`.
8. You have now trained an evaluator!
9. Specify the name in `Search-Algorithm.py`.
10. Run `main.py`.