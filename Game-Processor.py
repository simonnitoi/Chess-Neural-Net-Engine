import chess
import chess.pgn
import torch
import io
import zstandard as zstd

# SETTINGS
database_file_name = "lichess_db_standard_rated_2026-07.pgn.zst"
games_target = 1000
event_type = "rated rapid game"
rating_range = [1900,9999]
spawn_file_name = "Processed-Games.pt"

signal = input(f"Database -> {database_file_name}\nGames -> {games_target}\nEvent -> {event_type}\nRating -> ({rating_range[0]},{rating_range[1]})\nSpawn -> {spawn_file_name}\nEnter \"START\" to prep data -> ")

if signal != "START":
    exit()

def Convert_Position(board):
    tensor = torch.zeros((17,8,8), dtype=torch.uint8)

    if board.turn == chess.WHITE:

        # MY SIDE PIECES
        for square in board.pieces(chess.PAWN,chess.WHITE):
            tensor[0,chess.square_rank(square),chess.square_file(square)]=1
        
        for square in board.pieces(chess.KNIGHT,chess.WHITE):
            tensor[1,chess.square_rank(square),chess.square_file(square)]=1
        
        for square in board.pieces(chess.BISHOP,chess.WHITE):
            tensor[2,chess.square_rank(square),chess.square_file(square)]=1
        
        for square in board.pieces(chess.ROOK,chess.WHITE):
            tensor[3,chess.square_rank(square),chess.square_file(square)]=1

        for square in board.pieces(chess.QUEEN,chess.WHITE):
            tensor[4,chess.square_rank(square),chess.square_file(square)]=1

        for square in board.pieces(chess.KING,chess.WHITE):
            tensor[5,chess.square_rank(square),chess.square_file(square)]=1

        # OPPONENT SIDE PIECES
        for square in board.pieces(chess.PAWN,chess.BLACK):
            tensor[6,chess.square_rank(square),chess.square_file(square)]=1
            
        for square in board.pieces(chess.KNIGHT,chess.BLACK):
            tensor[7,chess.square_rank(square),chess.square_file(square)]=1
            
        for square in board.pieces(chess.BISHOP,chess.BLACK):
            tensor[8,chess.square_rank(square),chess.square_file(square)]=1
            
        for square in board.pieces(chess.ROOK,chess.BLACK):
            tensor[9,chess.square_rank(square),chess.square_file(square)]=1
                
        for square in board.pieces(chess.QUEEN,chess.BLACK):
            tensor[10,chess.square_rank(square),chess.square_file(square)]=1
                
        for square in board.pieces(chess.KING,chess.BLACK):
            tensor[11,chess.square_rank(square),chess.square_file(square)]=1

        # MY SIDE CASTLING
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[12,:,:]=1
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[13,:,:]=1

        # OPPONENT SIDE CASTLING
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[14,:,:]=1
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[15,:,:]=1

        # EN PASSANT
        if board.ep_square is not None:
            tensor[16,chess.square_rank(board.ep_square),chess.square_file(board.ep_square)]=1
        
    
    if board.turn == chess.BLACK:

        # MY SIDE PIECES
        for square in board.pieces(chess.PAWN,chess.BLACK):
            tensor[0,7-chess.square_rank(square),chess.square_file(square)]=1
                
        for square in board.pieces(chess.KNIGHT,chess.BLACK):
            tensor[1,7-chess.square_rank(square),chess.square_file(square)]=1
                
        for square in board.pieces(chess.BISHOP,chess.BLACK):
            tensor[2,7-chess.square_rank(square),chess.square_file(square)]=1
                
        for square in board.pieces(chess.ROOK,chess.BLACK):
            tensor[3,7-chess.square_rank(square),chess.square_file(square)]=1
        
        for square in board.pieces(chess.QUEEN,chess.BLACK):
            tensor[4,7-chess.square_rank(square),chess.square_file(square)]=1
        
        for square in board.pieces(chess.KING,chess.BLACK):
            tensor[5,7-chess.square_rank(square),chess.square_file(square)]=1
        
        # OPPONENT SIDE PIECES
        for square in board.pieces(chess.PAWN,chess.WHITE):
            tensor[6,7-chess.square_rank(square),chess.square_file(square)]=1
                    
        for square in board.pieces(chess.KNIGHT,chess.WHITE):
            tensor[7,7-chess.square_rank(square),chess.square_file(square)]=1
                    
        for square in board.pieces(chess.BISHOP,chess.WHITE):
            tensor[8,7-chess.square_rank(square),chess.square_file(square)]=1
                    
        for square in board.pieces(chess.ROOK,chess.WHITE):
            tensor[9,7-chess.square_rank(square),chess.square_file(square)]=1
                        
        for square in board.pieces(chess.QUEEN,chess.WHITE):
            tensor[10,7-chess.square_rank(square),chess.square_file(square)]=1
                        
        for square in board.pieces(chess.KING,chess.WHITE):
            tensor[11,7-chess.square_rank(square),chess.square_file(square)]=1

        # MY SIDE CASTLING
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[12,:,:]=1
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[13,:,:]=1

        # OPPONENT SIDE CASTLING
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[14,:,:]=1
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[15,:,:]=1

        # EN PASSANT
        if board.ep_square is not None:
            tensor[16,7-chess.square_rank(board.ep_square),chess.square_file(board.ep_square)]=1

    return tensor

# board = chess.Board("rnbqk2r/ppp2ppp/4pn2/8/2Pp4/3BPP2/PP1PN1PK/RNBQ1R2 b kq c3 0 7")

# print(board)
# tensor = Convert_Position(board)
# print(f"\nBoard checks\nShape -> {tensor.shape}")
# print(f"Type -> {tensor.dtype}")
# print(f"Pieces -> {tensor[:12].sum()}")
# print(f"My kC -> {tensor[12].sum()/64}")
# print(f"My qC -> {tensor[13].sum()/64}")
# print(f"Opponent kC -> {tensor[14].sum()/64}")
# print(f"Opponent qC -> {tensor[15].sum()/64}")
# print(f"En passant -> {tensor[16].sum()}")

def Convert_Outcome(result,turn):
    mod = 1
    if turn == chess.BLACK:
        mod = -1

    if result == "1/2-1/2":
        return 0

    if result == "1-0":
        return mod

    if result == "0-1":
        return -mod

tensor_list = []
outcome_list = []

with open(database_file_name,"rb") as compressed_file:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(compressed_file) as stream:
        text_stream = io.TextIOWrapper(stream, encoding="utf-8")

        # START CONVERTING
        games_processed = 0
        games_scanned = 0
        print("Beginning program...")
        while games_processed < games_target:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break

            w_elo = game.headers.get("WhiteElo")
            b_elo = game.headers.get("BlackElo")
            if w_elo == "?" or b_elo == "?":
                w_elo = -1
                b_elo = -1
            else:
                w_elo = int(w_elo)
                b_elo = int(b_elo)

            if (game.headers.get("Result") != "*") and (rating_range[0] < w_elo < rating_range[1]) and (rating_range[0] < b_elo < rating_range[1]) and (game.headers.get("Event").lower() == event_type.lower()):
                board = chess.Board()
                for move in game.mainline_moves():
                    tensor = Convert_Position(board)
                    outcome = Convert_Outcome(game.headers.get("Result"),board.turn)
                    tensor_list.append(tensor)
                    outcome_list.append(outcome)
                    board.push(move)
                games_processed += 1
            games_scanned += 1
            if games_scanned % 1000 == 0:
                print(f"Scanned -> {games_scanned} | Processed -> {games_processed}/{games_target}")

x = torch.stack(tensor_list)
y = torch.tensor(outcome_list,dtype=torch.float32)

torch.save({"tensors":x, "outcomes":y},spawn_file_name)

print(f"\n>>> PROGRAM FINISHED <<<\nSCANNED -> {games_scanned}\nPROCESSED -> {games_processed}/{games_target}")