import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim import Adam


class Upscaler(nn.Module):
    def __init__(self, upscale_factor=2, dtype=torch.float32):
        super(Upscaler, self).__init__()
        self.dtype = dtype
        self.upscale_factor = upscale_factor

        # Convolutional layer
        self.conv = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            padding=1,
            dtype=dtype
        )

        # Transposed convolutional layer for upsampling
        self.upconv = nn.ConvTranspose2d(
            16,
            3,
            kernel_size=2 * upscale_factor,
            stride=upscale_factor,
            padding=upscale_factor // 2,
            dtype=dtype
        )

        self.loss = MSELoss()
        self.optimizer = Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.upconv(x)
        return x

    def update(self, frame):
        frame = frame.permute(0, 3, 1, 2)

        smaller_size = list(frame.shape)
        smaller_size[2] //= 2
        smaller_size[3] //= 2
        smaller_frame = F.interpolate(frame, size=smaller_size[2:], mode="nearest")

        predict_frame = self.forward(smaller_frame)

        loss = self.loss(predict_frame, frame)
        if loss > 0.01:  # No overfitting?? maybe??
            loss.backward()

            # Update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Re-fill the stream with actual data.
        predict_frame[0, :, ::2, ::2] = smaller_frame
        predict_frame = predict_frame.permute(0, 2, 3, 1)

        return predict_frame, loss
