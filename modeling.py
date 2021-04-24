import torchvision
import torch.nn as nn
import copy
from utils import get_iou

class RND(nn.Module):
    """
    ResNetDetector
    Resnet50 backbone & two separate layers on top:
    One for object classification
    Another for relative coordinates regression  
    """
    def __init__(self):
        super().__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet1 = nn.Sequential(*list(resnet50.children())[:-3])
        self.resnet2 = nn.Sequential(*list(resnet50.children())[-3:-1])
        self.pooler = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(nn.Flatten(),
                                              nn.Linear(2048, 1, bias=True), 
                                              nn.Sigmoid())
        self.fpn = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                        nn.ReLU(inplace=True),
                                       )
        self.box_regressor = nn.Sequential(nn.Linear(3136, 4),
                                                 nn.Sigmoid())
        self.flattener = nn.Flatten()
        
    def forward(self, input_data):
        features = self.resnet1(input_data)
        fpn = self.fpn(features)
        fpn = self.pooler(fpn)
        fpn = self.flattener(fpn)
        features = self.resnet2(features)
        class_pred = self.classifier(features)
        coords = self.box_regressor(fpn)
        return class_pred, coords


def train_model(model, optimizer, train_dataloader, val_dataloader, criterion, regression_criterion, num_epochs=1):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mIoU = 0.0
    
    loss_history = []
    iou_history = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_iou = 0.0
        
        for inputs, labels in train_dataloader:
            
            inputs = inputs.to(device)
            target_bbox = labels[:, :4].to(device)
            target_class = labels[:, 4].to(device)
            
            optimizer.zero_grad()

            pred_class, pred_bbox = model(inputs)
            classification_loss = criterion(pred_class.squeeze(), target_class)
            
            regression_loss = regression_criterion(pred_bbox, target_bbox).sqrt()
            loss = 0.5 * classification_loss + regression_loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data
                    
        for inputs, labels in val_dataloader:
            
            inputs = inputs.to(device)
            target_bbox = labels[:, :4].to(device)
            target_class = labels[:, 4].to(device)
            
            with torch.no_grad():
                pred_class, pred_bbox = model(inputs)
            
            mIoU = get_iou(pred_bbox, target_bbox, inputs)
            
            running_iou += mIoU
            
        mean_loss = running_loss/len(train_dataloader)
        mIoU = running_iou/len(val_dataloader)

        loss_history.append(mean_loss)
        iou_history.append(mIoU)

        print(f'Epoch {epoch}/{num_epochs-1}. Loss: {mean_loss:.4f}. mIoU: {mIoU:.4f}')
        
        if mIoU > best_mIoU:
            best_mIoU = float(mIoU.cpu())
            best_model_wts = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_wts)
    return model, loss_history, iou_history