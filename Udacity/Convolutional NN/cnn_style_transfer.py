# @SEPARATING Style and Content

# CNN -> deeper layers understand more complicated features / content -> content representation of an image

# style transfer = content of image + style of another image

# texture + color of an image = spatial correlations

# correlation between each layer's feature maps

# content image + style image = new image

# @VGG19 and Content Loss

# step 1: pass both content and style image (CIM and SIM) throug the VGG19 network

# CIM -> content represtentation
# SIM -> style representation

# HOW TO COMBINE

# Defintion of Content in the NN = Output of Conv 4_2 layer

# Can define a loss between Content Representation of target and Content

# L_content  = (1/2) SUM (Tc - Cc)^2
# minimize this LOSS by changing the TARGET to be close the Content
# in other words CHANGE Tc to be near Cc

# (?)Minimize  L_Style = L_Content

# GRAM MATRIX
# d*h*w = gram matrix = "transpose"
# given d*h*w TRANSPOSE or GRAM = (h*w)*d
# correlation of "Features" found in t each layer

# 4*4 -> 4*4*8 (8 features maps)
# vectorize 4*4 = 16 (*8)
# 16*8 feature map -> get T(ranspose)
# f.T = 8*16 multiply

# (8 x 16) * (16 * 8) = the gram matrix
# each feature is treated as a non-localized non-connected value

# 8 by 8 GRAM MATRIX = similarity of the features / correlation of the features

# GRAM matrix is only ONE representation (there are other definitions of style)

# @STYLE LOSS

# L_style = a (SUM W_i(T_si-Ssi)^2)
# W = style weights (how much each "style" layer is affected

# L_style + L_content = Target Loss
# Minimize Target Loss only be change Tc and T_si

# ALPHA*L_content + BETA*L_style = Target Loss
# ALPHA / BETA : content / style ratio

# @VGG features

# @pretrained VGG19 

from PIL import Image
vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
	param.requires_grad(False) # make sure NOT to update weights during pass through / "training", Just feature extraction.

vgg.to (device) #move to GPU if available

''' Sequential ( (0): layer description...)'''

transform.Compose ([list of transformations])
content = load_image ('path/image').to(device)
style = load_image ('path/image', shape=content.shape[-2:]).to(device) 
# reshaping for GRAMS

# @get features functions

''' layers = {'0':'conv1_1',...} '''
#-> Look at Sequential above to complete the list
# NOTE: {'21': 'conv4_2'} content layer is a feature representation

# @gram matrix
_, d, h, w = tensor.size ()
tensor = tensor.view (d, h*w) #flatten torch to inverse
gram = torch.mm (tensor, tensor.t())

# matrix multiplication to get feature correlations
content_features = get_features (content_image, vgg)
style_features = get_features (style_image, vgg)
style_grams = {layer: gram_matrix[layer] for layer in style_features}

# get only the GRAMS i.e. CORRELATIONS for each style layer
target = content.clone().requires_grad_(True).to(device)

#requires_grad_ because you have to be able to manipulate the taget into the combined image

