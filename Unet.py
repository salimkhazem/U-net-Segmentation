import torch
from torch.nn import functional as F
import torch.nn as nn


def tensor_crop(tensor, target_tensor):
    """
    Using this function to crop the input tensor for the purpose to have the same
    size of the target tensor
    : Shape of tensor is like this : (Batch_size, Channel, Height, Width)
    :parameter tensor: tensor that need to be cropped
    :parameter target_tensor: target tensor which has smaller size
    :return : Cropped tensor
    """
    tensor_size = tensor.size()[2]
    target_size = target_tensor.size()[2]
    diff = tensor_size - target_size
    diff = diff // 2
    return tensor[:, :, diff : tensor_size - diff, diff : tensor_size - diff]


def double_conv(input_channels, output_channels):
    """
    This function applies two convolutional layers which are followed by  an activation
    function ReLu for this case
    :parameter input_channels : number of input channel
    :parameter output_channels : number of output channel
    :return a conv layer downed

     Note: for the implemetation we're using Sequential from nn to keep the order
    """

    conv = nn.Sequential(
        nn.Conv2d(
            input_channels, output_channels, kernel_size=3
        ),  # We are using a kernel (3x3)
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # we are using Max pooling
        self.max_pool_2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dconv_1 = double_conv(1, 64)
        self.dconv_2 = double_conv(64, 128)
        self.dconv_3 = double_conv(128, 256)
        self.dconv_4 = double_conv(256, 512)
        self.dconv_5 = double_conv(512, 1024)

        self.convTranspose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.convUp_1 = double_conv(1024, 512)

        self.convTranspose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.convUp_2 = double_conv(512, 256)

        self.convTranspose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.convUp_3 = double_conv(256, 128)

        self.convTranspose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.convUp_4 = double_conv(128, 64)

        self.output = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, image):

        # First Part : Encoder

        x1 = self.dconv_1(image)
        x2 = self.max_pool_2d(x1)
        x3 = self.dconv_2(x2)
        x4 = self.max_pool_2d(x3)
        x5 = self.dconv_3(x4)
        x6 = self.max_pool_2d(x5)
        x7 = self.dconv_4(x6)
        x8 = self.max_pool_2d(x7)
        x9 = self.dconv_5(x8)

        # Second Part : Decoder

        x = self.convTranspose_1(x9)
        y = tensor_crop(x7, x)
        x = self.convUp_1(torch.cat([x, y], axis=1))
        x = self.convTranspose_2(x)
        y = tensor_crop(x5, x)
        x = self.convUp_2(torch.cat([x, y], axis=1))
        x = self.convTranspose_3(x)
        y = tensor_crop(x3, x)
        x = self.convUp_3(torch.cat([x, y], axis=1))
        x = self.convTranspose_4(x)
        y = tensor_crop(x1, x)
        x = self.convUp_4(torch.cat([x, y], axis=1))

        # Final output
        output = self.output(x)
        print("Initial Input : ", x1.size())
        print("Final Output ", output.size())

        return output


if __name__ == "__main__":
    # For testing the Unet, we chose simple tensor of size (572*572) with c=1 and bs=1

    image = torch.rand((1, 1, 572, 572))
    model = UNet()
    print(model(image))
