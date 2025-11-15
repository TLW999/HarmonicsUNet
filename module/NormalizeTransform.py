import torch
import torch.nn as nn


class NormalizeTransform(nn.Module):
    def __init__(self):
        super(NormalizeTransform, self).__init__()
        self.div = 32768.0

    def forward(self, x):
        return x.float() / self.div


if __name__ == "__main__":
    normalize_module = NormalizeTransform()

    int_tensor = torch.randint(low=0, high=1000, size=(10,), dtype=torch.int16)
    normalized_tensor = normalize_module(int_tensor)

    print("Original integer tensor:", int_tensor)
    print("Normalized float tensor:", normalized_tensor)
