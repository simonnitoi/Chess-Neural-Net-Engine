import chess
import importlib
import torch

converter = importlib.import_module("Game-Processor")
architecture = importlib.import_module("Model-Trainer")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

model = architecture.ChessNet().to(device)

model.load_state_dict(torch.load("DeepRook.pt", map_location=device, weights_only=True))

model.eval()

def Get_Raw(board):
    if board.is_game_over():
        result = board.outcome()
        if result.winner is None:
            return (0,None)
        else:
            return (-1,None)

    with torch.no_grad():
        raw = model(converter.Convert_Position(board).float().unsqueeze(0).to(device))

    value = raw[0]
    policy = raw[1].squeeze(0)

    move_grades = {}

    for move in board.legal_moves:
        move_code = converter.Convert_Move(board.turn,move)
        move_grades[move] = policy[move_code].item()

    return (value.item(),max(move_grades, key=move_grades.get))