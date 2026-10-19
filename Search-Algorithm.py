import chess
import importlib
import torch
import random
import math
import time

converter = importlib.import_module("Game-Processor")
architecture = importlib.import_module("Model-Trainer")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

model = architecture.ChessNet().to(device)

model.load_state_dict(torch.load("DeepRook.pt", map_location=device, weights_only=True))

model.eval()

def norm(side,eval):
    if side == chess.BLACK:
        return eval*-1
    else:
        return eval

def Get_Raw(board, temperature=1.0):
    if board.is_game_over():
        result = board.outcome()
        if result.winner is None:
            return (0,None,None)
        else:
            return (-1,None,None)

    with torch.no_grad():
        raw = model(converter.Convert_Position(board).float().unsqueeze(0).to(device))

    value = raw[0]
    policy = raw[1].squeeze(0)

    if temperature == 0:
        move_grades = {}
        for move in board.legal_moves:
            move_code = converter.Convert_Move(board.turn,move)
            move_grades[move] = policy[move_code].item()
        return ( value.item() , max(move_grades,key=move_grades.get) , move_grades )

    else:
        probabilities = torch.softmax(policy / temperature, dim=0)
        move_grades = {}
        for move in board.legal_moves:
            move_code = converter.Convert_Move(board.turn,move)
            move_grades[move] = probabilities[move_code].item()
        tot = sum(list(move_grades.values()))
        for move in move_grades.keys():
            move_grades[move] = move_grades[move]/tot
        best_move = random.choices(list(move_grades.keys()), weights=list(move_grades.values()))[0]
        return ( value.item() , best_move , move_grades )

def PUCT(t, n, nP, grade):
    if n > 0:
        tn = t/n
    else:
        tn = 0
    return tn + 2*grade* ( math.sqrt( nP+1 ) / (n + 1) )

def MCTS_Step(board, data, limit=6, temperature=1.0):
    if data["status"] == "leaf":

        if board.is_game_over():
            data["status"] = "dead"
            valueR = Get_Raw(board,temperature)[0] # the function will catch it and return without running the model
            data["t"] -= valueR
            data["n"] += 1
            return data

        move_grades = Get_Raw(board,temperature)[2]
        for move in board.legal_moves:
            data["moves"][move] = {"t":0,"n":0,"grade":move_grades[move],"status":"leaf","moves":{}}
        data["status"] = "branch"

        move = Select(data,temperature)
        board.push(move)
        valueR = Get_Raw(board,temperature)[0]
        data["t"] += valueR
        data["n"] += 1
        data["moves"][move]["t"] = -valueR
        data["moves"][move]["n"] = 1

        board.pop()

        return data

    elif data["status"] == "branch":
        move = Select(data,temperature)
        board.push(move)

        old_t = data["moves"][move]["t"]
        data["moves"][move] = MCTS_Step(board , data["moves"][move] , limit , temperature)
        data["n"] += 1
        data["t"] -= data["moves"][move]["t"] - old_t

        board.pop()

        return data

    elif data["status"] == "dead":
        valueR = Get_Raw(board,temperature)[0] # the function will catch it and return without running the model
        data["t"] -= valueR
        data["n"] += 1
        return data


def Get_MCTS(board, limit=6, time_limit=5.0, temperature=1.0): # keep temperature at 1.0

    startTime = time.perf_counter()

    data = {"n":0,"moves":{}}

    move_grades = Get_Raw(board,temperature)[2]
    for move in board.legal_moves:
        data["moves"][move] = {"t":0,"n":0,"grade":move_grades[move],"status":"leaf","moves":{}}

    steps = 0
    while time.perf_counter()-startTime < time_limit:
        move = Select(data,temperature)
        board.push(move)

        data["moves"][move] = MCTS_Step(board , data["moves"][move] , limit , temperature)

        data["n"] += 1

        board.pop()
        steps+=1

    moves = list(data["moves"].keys())

    n=0
    highest = None
    for move in moves:
        if data["moves"][move]["n"] > n:
            n = data["moves"][move]["n"]
            highest = move

    return (data["moves"][highest]["t"]/n,highest,steps)


def Select(dict, temperature=1.0):
    max = -math.inf
    maxes = {}
    for move in dict["moves"].keys():
        puct = PUCT(dict["moves"][move]["t"],dict["moves"][move]["n"],dict["n"],dict["moves"][move]["grade"])
        if puct > max:
            max = puct
            maxes = {move : dict["moves"][move]["grade"]}
        elif puct == max:
            maxes[move] = dict["moves"][move]["grade"]

    maxes_list = list(maxes.keys())

    if len(maxes) == 1:
        return maxes_list[0]

    if temperature == 0:
        best = maxes_list[0]
        for move in maxes_list[1:]:
            if maxes[move] > maxes[best]:
                best = move
        return best

    else:
        return random.choices(list(maxes.keys()), weights=list(maxes.values()))[0]