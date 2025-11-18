import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Food101
import pytorch_lightning as pl
from timm import create_model


# ==============
# DataModule
# ==============

'''1) Load the Food101 datset
   2) Apply Transforms (data augmentations for training, resizing for validation)
   3) Return dataloaders with eht ecorrect batch size (images per patch), shuffling, and workers
   num_workers - background threads to load data
   persistent_workers - true means keeps worker threads alive which makes it faster'''

class Food101DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def setup(self,stage=None):
        self.train_dataset = Food101(
                                root=self.data_dir,
                                split='train',
                                transform=self.train_transform,
                                download=True)
        self.val_dataset = Food101(
                                root=self.data_dir,
                                split='test',
                                transform=self.val_transform,
                                download=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=True)

# ===============
# ViT model
# ===============

'''Loading a full ViT model (vit_small_patch16_224)
   Keeping on a subset of the transformer blocks
   Keeping all patch embedding and classication heads
   Running the forward pass just like ViT, but with fewer layers
   It is "reduced" ViT that uses fewer transformer blocks
   Choose how many blocks with num_blocks
   **smaller, faster ViT**
   uses only the first N layers of a standard ViT model
   while keeping the same classification head.
'''
class ViTLayerReduction(nn.Module):
    def __init__(self, num_blocks=10, num_classes=101):
        super().__init__()
        full_model = create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=num_classes,
            drop_rate=.5,
            drop_path_rate=0.2
        )
        #extracting important internal modules
        self.patch_embed = full_model.patch_embed #converts image patches into vectors
        self.cls_token = full_model.cls_token #special 'classification token'
        self.pos_embed = full_model.pos_embed #position embeddings
        self.pos_drop = full_model.pos_drop #drop out after adding positions
        #ViT normally has 12 blocks (ViT-small)  We keep 10 out of 12
        #self.blocks = nn.Sequential(*list(full_model.blocks[:num_blocks])) 
        '''nn.Sequential wraps modules and sometimes hides attributes, including num_heads
        nn.ModuleList preserves every block exactly as built.'''
        self.blocks = nn.ModuleList(list(full_model.blocks[:num_blocks]))
        #convert CLS token into class logits
        self.norm = full_model.norm
        self.head = full_model.head

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # ViT converts each 16x16 patch 384 dim vector
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1) 
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)  #pass through reduced transformer blocks
        x = self.norm(x)
        return self.head(x[:,0]) # the CLS token output that becomes logits over 101 classes.
        