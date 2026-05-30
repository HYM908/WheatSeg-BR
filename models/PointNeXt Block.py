class PointNeXtBlock(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.conv1 = nn.Conv1d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(hidden, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + identity)
