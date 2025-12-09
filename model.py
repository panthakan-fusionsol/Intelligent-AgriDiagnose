from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models;
import torch.nn as nn;


# Create ResNet Model
def create_resnet18_model(num_classes, pretrained=True):
    # Load pre-trained ResNet-18 (changed to ResNet-18 as requested)
    model = models.resnet18(pretrained=pretrained)
    
    # Modify the final layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def create_resnet34_model(num_classes, pretrained=True):
    # Load pre-trained ResNet-34
    model = models.resnet34(pretrained=pretrained)
    
    # Modify the final layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def create_resnet50_model(num_classes, pretrained=True):
    # Load pre-trained ResNet-50
    model = models.resnet50(pretrained=pretrained)
    
    # Modify the final layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def create_resnet101_model(num_classes, pretrained=True):
    # Load pre-trained ResNet-101
    model = models.resnet101(pretrained=pretrained)
    
    # Modify the final layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def create_efficientnet_b0_model(num_classes, pretrained=True):
    # Load pre-trained EfficientNet-B0
    if pretrained:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # Modify the classifier for our number of classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model


def create_efficientnet_v2_s_model(num_classes, pretrained=True):
    # Load pre-trained EfficientNet-V2-S
    if pretrained:
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_v2_s(weights=None)
    
    # Modify the classifier for our number of classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model