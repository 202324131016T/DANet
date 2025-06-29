

#commented out variables are handled by argparse in main.py
debug = True
# batch_size = 512
# num_workers = 0
# lr = 0.001
weight_decay = 1e-3
patience = 2
factor = 0.5
# epochs = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'resnet50'

# pre-train for image encoder
'''
"resnet50"
"resnet101"
"resnet152"
"vit_base_patch32_224"
"vit_large_patch32_224_in21k"
"vit_base_patch32_224_clip_laion2b"
"densenet121"
"densenet169"
"densenet161"
"densenet201"
'''

image_embedding = 2048
spot_embedding = 3467 #number of shared hvgs (change for each dataset)

# modify
pretrained = False
trainable = True 
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1


#  'vit_base_patch32_224',
#  'vit_base_patch32_224_clip_laion2b',
#  'vit_base_patch32_224_in21k',
#  'vit_base_patch32_224_sam',
