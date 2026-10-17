import torch
import torch.nn
import torch.nn.functional
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim

if __name__ == "__main__":
    # SETTINGS
    tensors_file_name = "Processed-Games.pt"
    epoch_target = 10
    model_file_name = "DeepRook.pt"

    signal = input(f"Tensors -> {tensors_file_name}\nEpochs -> {epoch_target}\nModel -> {model_file_name}\nEnter \"START\" to train model -> ")

    if signal != "START":
        exit()

class ChessNet(torch.nn.Module):

    def __init__(self):
        super(ChessNet, self).__init__()

        # SHARED
        self.cv1 = torch.nn.Conv2d(17,32,3,padding=1)
        self.cv2 = torch.nn.Conv2d(32,32,3,padding=1)
        self.cv3 = torch.nn.Conv2d(32,32,3,padding=1)
        self.cv4 = torch.nn.Conv2d(32,32,3,padding=1)

        # VALUE HEAD
        self.Vln1 = torch.nn.Linear(2048,64)
        self.Vln2 = torch.nn.Linear(64,1)

        # POLICY HEAD
        self.Pln1 = torch.nn.Linear(2048,256)
        self.Pln2 = torch.nn.Linear(256,4609)
        
    def forward(self,x):
        # SHARED
        x = torch.nn.functional.relu(self.cv1(x))
        x = torch.nn.functional.relu(self.cv2(x))
        x = torch.nn.functional.relu(self.cv3(x))
        x = torch.nn.functional.relu(self.cv4(x))
        x = torch.flatten(x, start_dim=1)

        # VALUE HEAD
        val = torch.nn.functional.relu(self.Vln1(x))
        val = torch.tanh(self.Vln2(val))

        # POLICY HEAD
        pol = torch.nn.functional.relu(self.Pln1(x))
        pol = self.Pln2(pol)

        return val,pol

if __name__ == "__main__":
    data = torch.load(tensors_file_name)
    x = data["tensors"]
    y = data["outcomes"]
    z = data["moves"]

    data_set = TensorDataset(x,y,z)

    training_amount = round(0.9*len(data_set))
    validation_amount = len(data_set)-training_amount

    training_set,validation_set = random_split(data_set, [training_amount, validation_amount])

    training_loader = DataLoader(dataset=training_set,batch_size=64,shuffle=True)
    validation_loader = DataLoader(dataset=validation_set,batch_size=64,shuffle=False)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model = ChessNet().to(device)

    value_criterion = torch.nn.MSELoss()
    policy_criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    epoch = 0
    while epoch < epoch_target:
        total_val_training_loss = 0
        total_pol_training_loss = 0
        for boards, outcomes, moves in training_loader:
            boards = boards.float().to(device)
            outcomes = outcomes.to(device).unsqueeze(1)
            moves = moves.to(device)

            optimizer.zero_grad()
            val_out,pol_out = model(boards)

            val_loss = value_criterion(val_out,outcomes)
            pol_loss = policy_criterion(pol_out,moves)
            loss = val_loss+pol_loss

            total_val_training_loss += val_loss.item()
            total_pol_training_loss += pol_loss.item()

            loss.backward()
            optimizer.step()
        with torch.no_grad():
            total_val_validation_loss = 0
            total_pol_validation_loss = 0
            for boards, outcomes, moves in validation_loader:
                boards = boards.float().to(device)
                outcomes = outcomes.to(device).unsqueeze(1)
                moves = moves.to(device)

                val_out,pol_out = model(boards)

                val_loss = value_criterion(val_out,outcomes)
                pol_loss = policy_criterion(pol_out,moves)
                loss = val_loss+pol_loss
                
                total_val_validation_loss += val_loss.item()
                total_pol_validation_loss += pol_loss.item()
        epoch+=1
        print(f"\nAverage Training Value Loss -> {total_val_training_loss/len(training_loader)}\nAverage Training Policy Loss -> {total_pol_training_loss/len(training_loader)}")
        print(f"Average Validation Value Loss -> {total_val_validation_loss/len(validation_loader)}\nAverage Validation Policy Loss -> {total_pol_validation_loss/len(validation_loader)}")
        print(f"Average Training Loss -> {(total_val_training_loss+total_pol_training_loss)/len(training_loader)}\nAverage Validation Loss -> {(total_val_validation_loss+total_pol_validation_loss)/len(validation_loader)}")
        print(f"Epochs -> {epoch}/{epoch_target}")

    torch.save(model.state_dict(), model_file_name)

    print(f"\n>>> PROGRAM FINISHED <<<")
    print(f"\nAVERAGE TRAINING VALUE LOSS -> {total_val_training_loss/len(training_loader)}\nAVERAGE TRAINING POLICY LOSS -> {total_pol_training_loss/len(training_loader)}")
    print(f"AVERAGE VALIDATION VALUE LOSS -> {total_val_validation_loss/len(validation_loader)}\nAVERAGE VALIDATION POLICY LOSS -> {total_pol_validation_loss/len(validation_loader)}")
    print(f"AVERAGE TRAINING LOSS -> {(total_val_training_loss+total_pol_training_loss)/len(training_loader)}\nAVERAGE VALIDATION LOSS -> {(total_val_validation_loss+total_pol_validation_loss)/len(validation_loader)}")
    print(f"EPOCHS -> {epoch}/{epoch_target}")