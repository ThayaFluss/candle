from torch.nn.modules.loss import MSELoss


class HalfMSELoss(MSELoss):
    def __init__(self, reduction="mean"):
        super().__init__(reduction=reduction)
        

    def forward(self, input, target):
        return  super().forward(input, target)/2