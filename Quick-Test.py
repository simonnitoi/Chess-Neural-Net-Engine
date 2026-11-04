import chess
import chess.pgn
import importlib

engine = importlib.import_module("Search-Algorithm")

board = chess.Board()

# << YOU PLAY IT >>
while True:
    playerSide = input("\"w\" or \"b\" -> ")
    if playerSide == "w" or playerSide == "b":
        break

if (playerSide == "b") and (board.turn == chess.WHITE):
    calc = engine.Get_MCTS(board)

    # print(f"\nEval -> {engine.norm(board.turn,calc[0])}\nSteps -> {calc[2]}\nMove -> {calc[1].uci()}\nFEN -> {board.fen()}")

    board.push(calc)

if (playerSide == "w") and (board.turn == chess.BLACK):
    calc = engine.Get_MCTS(board)

    # print(f"\nEval -> {engine.norm(board.turn,calc[0])}\nSteps -> {calc[2]}\nMove -> {calc[1].uci()}\nFEN -> {board.fen()}")

    board.push(calc)

while not board.is_game_over():
    print(f"{board}\n\nLegal Moves: {str(board.legal_moves).split('(')[1].split(')')[0]}\nType \"pgn\" to end game early\n")
    playerMove = input("Your Move -> ")
    if playerMove == "pgn":
        break
    while True:
        try:
            board.push_san(playerMove)
            break
        except:
            playerMove = input("Error! Your Move -> ")
    if not board.is_game_over():
        calc = engine.Get_MCTS(board)
        
        # print(f"\nEval -> {engine.norm(board.turn,calc[0])}\nSteps -> {calc[2]}\nMove -> {calc[1].uci()}\nFEN -> {board.fen()}")
        board.push(calc)



# << IT PLAYS ITSELF >>
# while not board.is_game_over():
#     board.push(engine.Get_MCTS(board,time_limit=1.0)[1])   # May take a while unless you time limit it to be really dumb
#     if board.is_game_over():
#         break
#     board.push(engine.Get_Raw(board)[1])



print(f"---------------\n\n{board}")
print(str(chess.pgn.Game.from_board(board)))
print("\n---------------")