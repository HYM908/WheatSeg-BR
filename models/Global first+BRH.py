class PointNeXtSemSeg(nn.Module):
    def __init__(self, num_classes=3, feature_channel=0, use_rgb=True):
        super().__init__()

        print("🔥 USING NEW PointNeXtSemSeg WITH feature_channel 🔥")
        print(f"🔥 feature_channel RECEIVED = {feature_channel}")
        print(f"🔥 use_rgb RECEIVED = {use_rgb}")
        # xyz always 3
        xyz_c = 3
        rgb_c = 3 if use_rgb else 0
        geo_c = int(feature_channel) if feature_channel is not None else 0

        in_channels = xyz_c + rgb_c + geo_c

        print(f"🔥 xyz_c={xyz_c}, rgb_c={rgb_c}, geo_c={geo_c} -> in_channels={in_channels}")

        print(f"🔥 COMPUTED in_channels = {in_channels}")
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        # Encoder
        self.stage1 = nn.Sequential(
            PointNeXtBlock(64),
            PointNeXtBlock(64)
        )

        assert self.stem[0].in_channels == in_channels, \
            f"stem in_channels mismatch: stem={self.stem[0].in_channels}, expected={in_channels}"

        self.trans1 = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU()
        )

        self.stage2 = nn.Sequential(
            PointNeXtBlock(128),
            PointNeXtBlock(128)
        )

        self.trans2 = nn.Sequential(
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU()
        )

        self.stage3 = nn.Sequential(
            PointNeXtBlock(256),
            PointNeXtBlock(256)
        )

        # Global
        self.global_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU()
        )

        # Coarse classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, num_classes, 1)
        )

        # Boundary prediction head（辅助任务）
        self.boundary_head = nn.Sequential(
            nn.Conv1d(128, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 1, 1)  # 输出 boundary logit
        )
        self.refine = BoundaryRefineHead(
            in_channels=128,
            hidden=128,
            num_classes=num_classes,
            k=16,
            layers=2
        )
