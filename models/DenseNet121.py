import torch.nn as nn
import torchvision.models as models


class CustomDenseNet121(nn.Module):
    def __init__(self, num_classes=9, dropout_rate=0.3):
        super(CustomDenseNet121, self).__init__()

        # Load the pre-built DenseNet-121 model
        self.model = models.densenet121(pretrained=True)
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Modify the model's last layer to match the number of classes (8) in the problem
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, num_classes),
            nn.Dropout(dropout_rate)
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_data):
        return self.model(input_data)

    def compute_loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)
