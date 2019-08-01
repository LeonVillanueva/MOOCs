'''
Step 0: Import Datasets

Make sure that you've downloaded the required human and dog datasets:

Note: if you are using the Udacity workspace, you DO NOT need to re-download these - they can be found in the /data folder as noted in the cell below.

    Download the dog dataset. Unzip the folder and place it in this project's home directory, at the location /dog_images.

    Download the human dataset. Unzip the folder and place it in the home directory, at location /lfw.

Note: If you are using a Windows machine, you are encouraged to use 7zip to extract the folder.

In the code cell below, we save the file paths for both the human (LFW) dataset and dog dataset in the numpy arrays human_files and dog_files.
'''

# LOOKUP : from glob import glob, glob module

'''
Before using any of the face detectors, it is standard procedure to convert the images to grayscale. The detectMultiScale function executes the classifier stored in face_cascade and takes the grayscale image as a parameter. 
'''

'''
In the above code, faces is a numpy array of detected faces, where each row corresponds to a detected face. Each detected face is a 1D array with four entries that specifies the bounding box of the detected face. The first two entries in the array (extracted in the above code as x and y) specify the horizontal and vertical positions of the top left corner of the bounding box. The last two entries in the array (extracted here as w and h) specify the width and height of the box.
'''

# @Haar Features: Similar to Convolutional Kernels / Layers

'''
1. Collect Negative Pictures (a lot > 1000s) 2x Positive
2. Collect Positive Pictures (a lot > 1000s)
3. Create Positive Vector File -> stitch with OpenCV
4. Train Cascade
'''

import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# HOW TO TRANSFORM JUST ONE IMAGE
# https://discuss.pytorch.org/t/how-to-classify-single-image-using-loaded-net/1411/2
imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

image = image_loader(PATH TO IMAGE)

your_trained_net(image)

# APPLICATION FINAL

from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable

pic = Image.open('test_dog.jpg')
trnf = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
img = trf(pic).float()
img = Variable(img, requires_grad=True)
img = img.unsqueeze(0)
pred = VGG16(img)


_, yhat = torch.max (pred, 1)

# DATA LOADER CELL1

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor()]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor()]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor()])
}
# DATA LOADER CELL2
        #~ Attribution: https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
directory = ('../data/dog_images')
    
image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(directory, x), 
        transform=data_transforms[x]
    )
    for x in ['train', 'valid', 'test']
}
# DATA LOADER CELL3
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

for x in ['train', 'valid', 'test']:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))

# @PRETRAINED MODEL PREDICTIONS

from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    
        #~ Load using module
    pic = Image.open(img_path) 
        #~ Define your transformations
    trnf = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
        #~ Apply transformations and run through VGG16
    img = trnf(pic).float()
    img = Variable(img, requires_grad=True)
    img = img.unsqueeze(0)
    if use_cuda:
        img = img.to ('cuda')
    pred = VGG16(img)
        #~ Prediction will be a list item (1), of possible images classifications, we only want the MAXIMUM, i.e.
        #~ maximum likelihood item (torch.max)
    _, yhat = torch.max (pred, 1)
        #~ Attribution: https://discuss.pytorch.org/t/how-to-classify-single-image-using-loaded-net/1411/2
        #~ We only want yhat which in this case is the Integer tensor. Then transform into a integer on return.
    return yhat.item()