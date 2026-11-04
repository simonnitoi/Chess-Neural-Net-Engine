# Chess Neural Net Engine
A simple PyTorch-based chess evaluator.

## How it Works
### Part 1: Tensorizing Data (`Game-Processor.py`)
1. Reads game data from a simple .pgn file, or .pgn.zst file via a piece-by-piece decompression stream so that the massive archive can stay compressed.
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
6. Once the requested number of games has been met, the positions and outcomes and moves are saved to a .pt file as uint8 and float32 and long respectively. This may be done in multiple steps/files.
### Part 2: Deep Learning (`Deep-Learning.py`)
1. The 17 initial tensor input channels are converted into 128 channels by a 2D convolutional layer, and then batch-normalized.
2. A core architecture is applied, combining residual connection and squeeze-and-excitation into 10 blocks, where each 2D convolutional layer is set to 128 in/out channels.
3. It is then split into a value and a policy head:
    - The value head pushes the 128 into 16 channels, batch normalizing, and then applying two linear layers to push the evaluation to a single value. Tanh is then applied to it.
    - The policy head pushes the 128 into 32 channels, batch normalizing, and then applying one linear layer to push the evaluation to score every value in the move value range.
4. The value and policy are outputted as a tuple.
5. The list of tensor files is looped through for the specified number of epochs:
    - Positions are split according to `random_split()` with a seed so that validation sets are all always unseen even when revisited in the loop.
    - The tensors are loaded with a batch size of 2048.
    - The model is optimized via AdamW with learn rate 0.001 and weight decay 0.0001.
    - Loss is done by MSELoss for the value head and CrossEntropyLoss for the policy head.
    - CosineAnnealingLR is applied to the learn rate of AdamW throughout the model's training.
    - Every position is graded according to the outcome of the game it is from and the move that was played next, for value and policy respectively.
6. The model is saved to a .pt file.
### Part 3: Monte Carlo Tree Search (`Search-Algorithm.py`)
1. It is applied into PUCT MCTS in `Search-Algorithm.py`, following the search formula `-tn + 2*g*math.sqrt(nP+1) / (n+1)`, where t = total evaluation score, n = number of visits, nP = number of parent node visits, and g is the model's assigned probability for the node (0 to 1).
### Part 4 Application (`Quick-Test.py` & `Run-Engine.bat`):
1. Run `Quick-Test.py` if you want to test it quickly.
2. Point a GUI such as Cute Chess to `Run-Engine.bat` to play it nicely.

## Steps to Replicate & Play the Model (beware of hardware constraints)
1. Clone the repository.
2. Install the `chess` library, `zstandard`, and PyTorch. (pip install chess zstandard torch)
3. Download a .pgn or .pgn.zst game archive.
4. Specify the archive file name, and your desired settings in `Game-Processor.py`, via the variables featured at the top.
5. Run `Game-Processor.py` (beware of RAM constraints).
6. Specify the tensor file name, and your desired settings in `Deep-Learning.py`, via the variables featured at the top.
7. Run `Deep-Learning.py`.
8. You have now trained an evaluator!
9. Specify the name in `Search-Algorithm.py`.
10. Run `Quick-Test.py` to test it.
11. To run via a GUI such as Cute Chess, specify the file pathing in `Run-Engine.bat` and point the GUI to this file.