class BoundaryRefineHead(nn.Module):
       def __init__(self, in_channels, hidden, num_classes, k=16, layers=2):
        super().__init__()
        self.k = k
        self.layers = layers

        self.edge1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        )

        if layers == 2:
            self.edge2 = nn.Sequential(
                nn.Conv2d(hidden * 2, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.GELU()
            )

        self.proj = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden, num_classes, 1)
        )

    def forward(self, feat_bcn, xyz_b3n):
        B, C, N = feat_bcn.shape

        feat = feat_bcn.transpose(1, 2).contiguous()  
        xyz = xyz_b3n.transpose(1, 2).contiguous()     

        idx = knn_indices(xyz, self.k)                  
        neigh = gather_neighbors(feat, idx)             
        center = feat.unsqueeze(2).expand_as(neigh)

        edge = torch.cat([center, neigh - center], dim=-1)  
        edge = edge.permute(0, 3, 1, 2).contiguous()         

        x = self.edge1(edge).max(dim=-1)[0]            

        if self.layers == 2:
            feat2 = x.transpose(1, 2).contiguous()       
            neigh2 = gather_neighbors(feat2, idx)
            center2 = feat2.unsqueeze(2).expand_as(neigh2)
            edge2 = torch.cat([center2, neigh2 - center2], dim=-1)
            edge2 = edge2.permute(0, 3, 1, 2).contiguous()
            x = self.edge2(edge2).max(dim=-1)[0]

        logits_res = self.proj(x)                
        return logits_res
