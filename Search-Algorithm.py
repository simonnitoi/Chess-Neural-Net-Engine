import chess
import importlib
import torch
import random
import math
import time
import sys

converter = importlib.import_module("Game-Processor")
architecture = importlib.import_module("Deep-Learning")

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

model = architecture.ChessNet().to(device)

model.load_state_dict(torch.load("Heisenberg.pt", map_location=device, weights_only=True))

model.eval()

def Get_Raw(board):
    if board.is_game_over():
        result = board.outcome()
        if result is None or result.winner is None:
            return (0,None,None,None)
        else:
            return (-1.5,None,None,None)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                raw = model(converter.Convert_Position(board).float().unsqueeze(0).to(device))
        else:
            raw = model(converter.Convert_Position(board).float().unsqueeze(0).to(device))

    value = raw[0].float()
    policy = torch.softmax(raw[1].squeeze(0).float(), dim=0).cpu().tolist()

    legal_policy = {}
    total = 0

    for move in board.legal_moves:
        move_grade = policy[converter.Convert_Move(board.turn,move)]

        total += move_grade

        legal_policy[move] = move_grade

    mod = 1/total
    probabilities = {}
    for move in legal_policy.keys():
        probabilities[move] = legal_policy[move]*mod

    eval = value.item()
    top_move = max(probabilities, key=probabilities.get)
    chance_move = random.choices( list(probabilities.keys()) , weights=list(probabilities.values()) )[0]

    if board.is_repetition(2):
        return 0 , top_move , chance_move , probabilities

    return eval , top_move , chance_move , probabilities




def PUCT(node , nP):
    t = node.t
    n = node.n
    g = node.grade

    if n > 0:
        tn = t/n
    else:
        tn = 0

    return -tn + 2*g*math.sqrt(nP+1) / (n+1)

class Node:
    def __init__(self,board,move,grade,type="leaf"):
        self.type = type
        self.board = board
        self.move = move

        self.grade = grade

        self.children = []
        self.eval = None

        self.t = 0
        self.n = 0

        if type == "base":
            calc = Get_Raw(self.board)

            for move in self.board.legal_moves:
                self.children.append(Node( self.board , move , calc[3][move] ))

        

    def Step(self):

        if self.type == "base":

            selected_node = self.Select()
            selected_node.Step()

            self.n +=1

        elif self.type == "dead":
            self.t += self.eval
            self.n +=1
            return self.eval


        elif self.type == "leaf":

            self.board.push(self.move)

            calc = Get_Raw(self.board)
            self.eval = calc[0]

            self.t += self.eval

            if self.board.is_game_over():
                self.type = "dead"

            else:
                self.type = "branch"

                for move in self.board.legal_moves:
                    self.children.append(Node( self.board , move , calc[3][move] ))

            self.n +=1
            self.board.pop()

            return self.eval


        else:

            self.board.push(self.move)

            selected_node = self.Select()

            step_eval = -selected_node.Step()
            self.t += step_eval

            self.n +=1
            self.board.pop()

            return step_eval

    def Select(self):
        top = None
        for node in self.children:
            score = PUCT(node , self.n)
            if top is None or top["score"] < score:
                top = {}
                top["score"] = score
                top["node"] = node

        return top["node"]

            

def Get_MCTS(board,time_limit=8):

    start_time = time.perf_counter()

    if board.is_game_over():
        return None

    if len(list(board.legal_moves)) == 1:
        return list(board.legal_moves)[0]

    base = Node(board,None,None,type="base")

    steps = 0
    while time.perf_counter()-start_time < time_limit:
        base.Step()
        steps+=1
    print(f"info string Steps Calculated -> {steps}")

    top_node = None
    for node in base.children:
        if top_node is None or node.n > top_node.n:
            top_node = node

    return top_node.move


def uci_loop():
    board = chess.Board()

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        line = line.strip()
        if not line:
            continue

        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            print("id name Heisenberg")
            print("id author Simon Nitoi")
            print("uciok")
            sys.stdout.flush()

        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()

        elif cmd == "ucinewgame":
            board.reset()

        elif cmd == "position":
            if "startpos" in line:
                board.reset()
                if "moves" in line:
                    moves_str = line.split("moves ")[1]
                    for move_uci in moves_str.split():
                        board.push_uci(move_uci)
            elif "fen" in line:
                fen_part = line.split("fen ")[1]
                if "moves" in fen_part:
                    fen, moves_str = fen_part.split(" moves ")
                    board.set_fen(fen)
                    for move_uci in moves_str.split():
                        board.push_uci(move_uci)
                else:
                    board.set_fen(fen_part)

        elif cmd == "go":
            top_move = Get_MCTS(board)

            best_move = top_move if top_move else list(board.legal_moves)[0]

            print(f"bestmove {best_move.uci()}")
            sys.stdout.flush()

        elif cmd == "quit":
            break

if __name__ == "__main__":
    uci_loop()