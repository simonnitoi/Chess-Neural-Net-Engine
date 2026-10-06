import torch
import torch.nn
import torch.nn.functional
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim

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
        self.cv1 = torch.nn.Conv2d(17,32,3,padding=1)
        self.cv2 = torch.nn.Conv2d(32,32,3,padding=1)
        self.cv3 = torch.nn.Conv2d(32,32,3,padding=1)
        self.ln1 = torch.nn.Linear(2048,64)
        self.ln2 = torch.nn.Linear(64,1)
        
    def forward(self,x):
        x = torch.nn.functional.relu(self.cv1(x))
        x = torch.nn.functional.relu(self.cv2(x))
        x = torch.nn.functional.relu(self.cv3(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.nn.functional.relu(self.ln1(x))
        x = torch.tanh(self.ln2(x))
        return x

data = torch.load(tensors_file_name)
x = data["tensors"]
y = data["outcomes"]

data_set = TensorDataset(x,y)

training_amount = round(0.9*len(data_set))
validation_amount = len(data_set)-training_amount

training_set,validation_set = random_split(data_set, [training_amount, validation_amount])

training_loader = DataLoader(dataset=training_set,batch_size=64,shuffle=True)
validation_loader = DataLoader(dataset=validation_set,batch_size=64,shuffle=False)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
model = ChessNet().to(device)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epoch = 0
while epoch < epoch_target:
    total_training_loss = 0
    for boards, outcomes in training_loader:
        boards = boards.float().to(device)
        outcomes = outcomes.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(boards)
        loss = criterion(outputs, outcomes)
        total_training_loss += loss.item()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        total_validation_loss = 0
        for boards, outcomes in validation_loader:
            boards = boards.float().to(device)
            outcomes = outcomes.to(device).unsqueeze(1)
            outputs = model(boards)
            loss = criterion(outputs, outcomes)
            total_validation_loss += loss.item()
    epoch+=1
    print(f"Average Training Loss -> {total_training_loss/len(training_loader)} | Average Validation Loss -> {total_validation_loss/len(validation_loader)} | Epochs -> {epoch}/{epoch_target}")

torch.save(model.state_dict(), model_file_name)

print(f"\n>>> PROGRAM FINISHED <<<\nAVERAGE TRAINING LOSS -> {total_training_loss/len(training_loader)} | AVERAGE VALIDATION LOSS -> {total_validation_loss/len(validation_loader)} | EPOCHS -> {epoch}/{epoch_target}")