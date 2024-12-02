import torch

class DeepNeuralNetworkModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_hidden_layers=2):
        super(DeepNeuralNetworkModel, self).__init__()

        # List of layers
        layers = []

        # Input layer
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(torch.nn.ReLU())  # ReLU activation after input layer

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())  # ReLU activation for each hidden layer

        # Output layer
        layers.append(torch.nn.Linear(hidden_dim, 1))  # Output layer for regression

        # Combine all layers into a sequential model
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)