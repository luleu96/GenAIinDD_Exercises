import torch

class DeepNeuralNetworkModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_hidden_layers=2):
        super(DeepNeuralNetworkModel, self).__init__()

        # Input layer
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )

        # Hidden layers: each has Linear + ReLU
        self.hidden_layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
            for _ in range(num_hidden_layers - 1)
        ])

        # Output layer
        self.output_layer = torch.nn.Linear(hidden_dim, 1)
        
    
    def forward(self, x):
        hidden_states = []  # List to store outputs from each layer
        
        # Input layer
        x = self.input_layer(x)
        hidden_states.append(x)  # Save output
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            hidden_states.append(x)  # Save output

        # Output layer
        x = self.output_layer(x)
        return x, hidden_states  # Return final output and intermediate states