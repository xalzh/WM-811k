import torch.nn as nn
from torchvision.models import mobilenet_v2


class CustomMobileNet(nn.Module):
    def __init__(self, num_classes=9, dropout_rate=0.3):
        super(CustomMobileNet, self).__init__()
        self.mobilenet = mobilenet_v2(pretrained=True)
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, 9)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_data):
        out = self.mobilenet(input_data)
        out = self.dropout(out)
        out = self.softmax(out)
        return out

    def compute_loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)
