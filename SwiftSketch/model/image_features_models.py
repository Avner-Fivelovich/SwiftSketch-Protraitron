
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import CLIP_.clip as clip
from transformers import  AutoImageProcessor, AutoModel


class DINO2Feutures(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-giant').to(device)
        self.device=device
        self.transform = transforms.Compose([
        transforms.ToTensor(),
    ])
        
    def image_preprocessing(self,image):
        image=image.convert("RGB")
        image = self.transform(image) 
        image = image.to(self.device)
        image=image.unsqueeze(0)
        return image
        

    def get_dino_features(self, image):
        inputs = self.dino_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.dino_model(**inputs)
        # outputs = outputs.last_hidden_state[:, 1:, :].mean(dim=1) #take mean of everything but cls
        # outputs = outputs.last_hidden_state[:,0,:] # take only cls
        outputs = outputs.last_hidden_state.mean(dim=1)   
        outputs /= outputs.norm(dim=-1, keepdim=True) 
        return outputs

    def forward(self, image):
        image= self.image_preprocessing(image)
        with torch.no_grad():
            features = self.get_dino_features(image)
            return features[0]
        

 
        
class CLIPMidlleFeutures(nn.Module):
    def __init__(self, device, layer):
        super().__init__()
        self.device = device
        self.clip_model, clip_preprocess = clip.load("RN101", device=device, jit=False)
        if device.type == "mps":
            self.clip_model = self.clip_model.float()
        # Freeze the CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.visual_model = self.clip_model.visual
        layers = list(self.visual_model.children())
        self.init_layers = torch.nn.Sequential(*layers[:8])
        self.layer1 = layers[8]
        self.layer2 = layers[9]
        self.layer3 = layers[10]
        self.layer4 = layers[11]
        self.att_pool2d = layers[12]

       
        self.middle_layer_index = layer 
        # clip normalization
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ]) 

        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                x = m.relu(bn(conv(x)))
            x = m.avgpool(x)
            return x

        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]

    def get_clip_features_from_middle_layer(self, images):
        images = torch.stack([self.normalize_transform(img) for img in images]).to(self.device)
        fc_features, features_list = self.forward_inspection_clip_resnet(images)
        if self.middle_layer_index<=4:
            return features_list[self.middle_layer_index]
        else:
            return fc_features
    
    def image_preprocessing(self,image):
        image=image.convert("RGB")
        image = self.transform(image) 
        image = image.to(self.device)
        image=image.unsqueeze(0)
        return image
    
    def forward(self, x):
        x= self.image_preprocessing(x)
        with torch.no_grad():
            features = self.get_clip_features_from_middle_layer(x)
        features = features.to(torch.float32)  
        return features[0]
        

       