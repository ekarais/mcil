import torch

class Protonet(torch.nn.Module):
    
    def __init__(self, **kwargs):
        super(Protonet, self).__init__()
        
        self.encoder = torch.nn.Sequential(
                            conv_block(1, kwargs["hid_dim"]),
                            conv_block(kwargs["hid_dim"], kwargs["hid_dim"]),
                            conv_block(kwargs["hid_dim"], kwargs["hid_dim"]),
                            conv_block(kwargs["hid_dim"], kwargs["z_dim"]),
                            torch.nn.Flatten())
    def forward(self, x):
        return self.encoder(x)

    def loss(self, )

def conv_block(in_channels, out_channels):
        return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2))