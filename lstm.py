import os
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, init_weights=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.softmax = nn.Softmax(dim=1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, prev_state=None):
        if prev_state == None:
            # Set initial states
            h_prev = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c_prev = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        else:
            h_prev, c_prev = prev_state

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, (h_prev, c_prev))
        # out = (batch_size, seq_length, n_directions*hidden_size)
        # h = (n_directions*n_layers, batch_size, hidden_size)
        # c = (n_directions*n_layers, batch_size, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out, (h, c)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)

def lstm(input_size, hidden_size, num_layers, num_classes, pretrained=False):
    if pretrained:
        raise NotImplementedError
    return LSTM(input_size, hidden_size, num_layers, num_classes)