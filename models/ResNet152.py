import torch.nn as nn
import torch
import torchvision.models as models


class CustomResNet152(nn.Module):
    def __init__(self, num_classes=9, dropout_rate=0.3):
        super().__init__()
        self.resnet = models.resnet152(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_data):
        out = self.resnet(input_data)
        out = self.dropout(out)
        return out

    def compute_loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)
