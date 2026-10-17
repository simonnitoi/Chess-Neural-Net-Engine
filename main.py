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

if playerSide == "b":
    calc = engine.Get_Raw(board)
    print(f"\nEval -> {calc[0]}")
    board.push(calc[1])

while not board.is_game_over():
    print(f"{board}\n\nLegal Moves: {str(board.legal_moves).split('(')[1].split(')')[0]}\n")
    playerMove = input("Your Move -> ")
    while True:
        try:
            board.push_san(playerMove)
            break
        except:
            playerMove = input("Error! Your Move -> ")
    if not board.is_game_over():
        calc = engine.Get_Raw(board)

        if board.turn == chess.BLACK:
            print(f"\nEval -> {calc[0]*-1}")
        else:
            print(f"\nEval -> {calc[0]}")

        board.push(calc[1])



# << IT PLAYS ITSELF >>
# while not board.is_game_over():
#     board.push(engine.Get_Raw(board)[1])



print(f"---------------\n\n{board}")
print(str(chess.pgn.Game.from_board(board)))
print("\n---------------")