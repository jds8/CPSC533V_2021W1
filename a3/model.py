import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(MyModel, self).__init__()
        # TODO YOUR CODE HERE FOR INITIALIZING THE MODEL

    def forward(self, x):
        # TODO YOUR CODE HERE FOR THE FORWARD PASS
        raise NotImplementedError()

    def select_action(self, state):
        self.eval()
        x = self.forward(state)
        self.train()
        return x.max(1)[1].view(1, 1).to(torch.long)
