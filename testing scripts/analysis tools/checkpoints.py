import torch

# Load the checkpoint
checkpoint = torch.load('model.pth')

# Print shapes of relevant tensors
if 'fc.weight' in checkpoint:
    fc_weight_shape = checkpoint['fc.weight'].shape
    print("Shape of fc.weight:", fc_weight_shape)
if 'fc.bias' in checkpoint:
    fc_bias_shape = checkpoint['fc.bias'].shape
    print("Shape of fc.bias:", fc_bias_shape)