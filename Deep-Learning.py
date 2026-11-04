import torch
import torch.nn
import torch.nn.functional
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim
import gc

if __name__ == "__main__":
    # SETTINGS
    tensor_file_names = ["Processed-Games-1.pt","Processed-Games-2.pt","Processed-Games-3.pt","Processed-Games-4.pt","Processed-Games-5.pt"]
    epoch_target = 3
    model_file_name = "Heisenberg.pt"

    signal = input(f"Tensors -> {tensor_file_names}\nEpochs -> {epoch_target}\nModel -> {model_file_name}\nEnter \"START\" to train model -> ")

    if signal != "START":
        exit()

class SEResBlock(torch.nn.Module):

    def __init__(self):
        super(SEResBlock, self).__init__()
        self.channels = 128
        self.se_channels = 32

        self.cv1 = torch.nn.Conv2d(self.channels,self.channels,3,padding=1,bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.channels)
        self.cv2 = torch.nn.Conv2d(self.channels,self.channels,3,padding=1,bias=False)
        self.bn2 = torch.nn.BatchNorm2d(self.channels)

        # SE layers
        self.se_fc1 = torch.nn.Linear(self.channels, self.se_channels)
        self.se_fc2 = torch.nn.Linear(self.se_channels, self.channels * 2)

    def forward(self,x):
        identity = x
        x = torch.nn.functional.relu(self.bn1(self.cv1(x)))
        x = self.bn2(self.cv2(x))


        # SE layers
        se = x.mean(dim=(2,3))

        se = torch.nn.functional.relu(self.se_fc1(se))
        se = self.se_fc2(se)

        se = se.unsqueeze(-1).unsqueeze(-1)

        gamma, beta = se.chunk(2, dim=1)

        x = torch.sigmoid(gamma) * x + beta


        x = x + identity
        x = torch.nn.functional.relu(x)
        return x

class ChessNet(torch.nn.Module):

    def __init__(self):
        super(ChessNet, self).__init__()
        self.blocks = 10

        # SHARED
        self.cv1 = torch.nn.Conv2d(17,128,3,padding=1)
        self.bn1 = torch.nn.BatchNorm2d(128)

        self.layers = torch.nn.ModuleList()
        for i in range(self.blocks):
            self.layers.append(SEResBlock())

        # VALUE HEAD
        self.Vcv1 = torch.nn.Conv2d(128,16,3,padding=1)
        self.Vbn1 = torch.nn.BatchNorm2d(16)

        self.Vln1 = torch.nn.Linear(1024,64)
        self.Vln2 = torch.nn.Linear(64,1)

        # POLICY HEAD
        self.Pcv1 = torch.nn.Conv2d(128,32,3,padding=1)
        self.Pbn1 = torch.nn.BatchNorm2d(32)
        self.Pln1 = torch.nn.Linear(2048,4609)
        
    def forward(self,x):

        # SHARED
        x = torch.nn.functional.relu(self.bn1(self.cv1(x)))

        for layer in self.layers:
            x = layer(x)

        # VALUE HEAD
        val = torch.nn.functional.relu(self.Vbn1(self.Vcv1(x)))
        val = torch.flatten(val, start_dim=1)
        val = torch.nn.functional.relu(self.Vln1(val))
        val = torch.tanh(self.Vln2(val))

        # POLICY HEAD
        pol = torch.nn.functional.relu(self.Pbn1(self.Pcv1(x)))
        pol = torch.flatten(pol, start_dim=1)
        pol = self.Pln1(pol)

        return val,pol

if __name__ == "__main__":

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model = ChessNet().to(device)

    value_criterion = torch.nn.MSELoss()
    policy_criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    scheduler = None

    epoch = 0

    while epoch < epoch_target:

        for file_name in tensor_file_names:

            print(f"\nStarted File -> {file_name} ({tensor_file_names.index(file_name)+1}/{len(tensor_file_names)})")
            
            data = torch.load(file_name)
            x = data["tensors"]
            y = data["outcomes"]
            z = data["moves"]
            data_set = TensorDataset(x,y,z)
            
            training_amount = round(0.9*len(data_set))
            validation_amount = len(data_set)-training_amount
            
            training_set,validation_set = random_split(data_set, [training_amount, validation_amount], generator=torch.Generator().manual_seed(42))
            
            training_loader = DataLoader(dataset=training_set,batch_size=2048,shuffle=True)
            validation_loader = DataLoader(dataset=validation_set,batch_size=2048,shuffle=False)
            
            if scheduler is None:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= len(tensor_file_names) * epoch_target * len(training_loader))

            tracker = 0

            total_val_training_loss = 0
            total_pol_training_loss = 0
            model.train()
            for boards, outcomes, moves in training_loader:
                boards = boards.float().to(device)
                outcomes = outcomes.to(device).unsqueeze(1)
                moves = moves.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                    val_out,pol_out = model(boards)

                    val_loss = value_criterion(val_out,outcomes)
                    pol_loss = policy_criterion(pol_out,moves)
                    loss = val_loss+pol_loss

                total_val_training_loss += val_loss.item()
                total_pol_training_loss += pol_loss.item()

                loss.backward()
                optimizer.step()

                tracker+=1
                if tracker % 200 == 0:
                    print(f"Trained Batch -> {tracker}")

                scheduler.step()

            tracker = 0
            
            with torch.no_grad():
                total_val_validation_loss = 0
                total_pol_validation_loss = 0
                model.eval()
                for boards, outcomes, moves in validation_loader:
                    boards = boards.float().to(device)
                    outcomes = outcomes.to(device).unsqueeze(1)
                    moves = moves.to(device)

                    with torch.amp.autocast(device.type, dtype=torch.bfloat16):

                        val_out,pol_out = model(boards)

                        val_loss = value_criterion(val_out,outcomes)
                        pol_loss = policy_criterion(pol_out,moves)
                        loss = val_loss+pol_loss
                        
                    total_val_validation_loss += val_loss.item()
                    total_pol_validation_loss += pol_loss.item()

                    tracker+=1
                    if tracker % 200 == 0:
                        print(f"Validated Batch -> {tracker}")

            print(f"\n>>> MODEL AUTOSAVE CHECKPOINT <<<")
            print(f"\nFinished File -> {file_name} ({tensor_file_names.index(file_name)+1}/{len(tensor_file_names)})")
            print(f"Average Training Value Loss -> {total_val_training_loss/len(training_loader)}\nAverage Training Policy Loss -> {total_pol_training_loss/len(training_loader)}")
            print(f"Average Validation Value Loss -> {total_val_validation_loss/len(validation_loader)}\nAverage Validation Policy Loss -> {total_pol_validation_loss/len(validation_loader)}")
            print(f"Average Training Loss -> {(total_val_training_loss+total_pol_training_loss)/len(training_loader)}\nAverage Validation Loss -> {(total_val_validation_loss+total_pol_validation_loss)/len(validation_loader)}")
            torch.save(model.state_dict(), model_file_name)

            del data, x, y, z, data_set, training_amount, validation_amount, training_set, validation_set, training_loader, validation_loader
            gc.collect()

        epoch+=1
        print(f"\n>>> EPOCH {epoch} OF {epoch_target} FINISHED <<<")

    print(f"\n>>> PROGRAM FINISHED <<<")