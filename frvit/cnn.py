import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


class LandmarkCNN(nn.Module):
    def __init__(self, num_landmarks):
        super().__init__()
        self.mobilenet = mobilenet_v3_small(pretrained=True)
        self.fc = nn.Linear(1000, num_landmarks * 2)

    def forward(self, x):
        x = self.mobilenet(x)
        x = self.fc(x)
        return x.view(x.shape[0], -1, 2)  # reshape to (batch_size, num_landmarks, 2)


# Define constants
NUM_LANDMARKS = 68  # for example
BATCH_SIZE = 16
CHANNELS = 3
HEIGHT = 112
WIDTH = 112

# Create model
model = LandmarkCNN(NUM_LANDMARKS)

# Create a random tensor with the correct size
img = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)

# Pass the input through the model
output = model(img)

# Print the output shape
print(output.shape)  # Should be [BATCH_SIZE, NUM_LANDMARKS * 2]
