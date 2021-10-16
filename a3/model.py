import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(MyModel, self).__init__()

        hidden_size = 8
        output_size = action_size
        self.neural_network = nn.Sequential(
            nn.Linear(state_size, hidden_size, bias=True, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=True, dtype=torch.double),
        )

    def forward(self, state):
        if type(state) != torch.Tensor:
            return self.neural_network(torch.from_numpy(state).type(torch.double)).type(torch.double)
        else:
            return self.neural_network(state.type(torch.double)).type(torch.double)

    def select_action(self, state):
        self.eval()
        x = self.forward(state)
        self.train()
        return x.max(1)[1].view(1, 1).to(torch.long)
