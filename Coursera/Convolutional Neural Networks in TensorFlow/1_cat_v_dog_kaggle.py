# !pip install tensorflow==2.0.0-alpha0 

# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb
# visualize through layers API
# plotting history

''' # Note Definitely not my code: 

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# -----------------------------------------------------------------------
# Now let's display our representations
# -----------------------------------------------------------------------
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  
  if len(feature_map.shape) == 4:
    
    #-------------------------------------------
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    #-------------------------------------------
    n_features = feature_map.shape[-1]  # number of features in the feature map
    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
    
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    
    #-------------------------------------------------
    # Postprocess the feature to be visually palatable
    #-------------------------------------------------
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std ()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

    #-----------------
    # Display the grid
    #-----------------

    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 

'''

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    source_list = os.listdir (SOURCE)
    clean_source = [s for s in source_list if os.path.getsize ( os.path.join ( SOURCE, s ) ) > 0] # checks for ZERO
    train_list = random.sample ( clean_source, int ( len(clean_source)*SPLIT_SIZE ) )
    test_list = list(set(clean_source) - set(train_list))
    
    for i in train_list:
        copyfile ( SOURCE+i, TRAINING+i )
    for i in test_list:
        copyfile ( SOURCE+i, TESTING+i )

l = [
            tf.keras.layers.Conv2D (16, 3, input_shape=(i, j, ch), activation='relu'),
            tf.keras.layers.MaxPooling2D (2, 2),
            tf.keras.layers.Conv2D (32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D (2, 2),
            tf.keras.layers.Conv2D (64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D (2, 2),
            tf.keras.layers.Flatten (),
            tf.keras.layers.Dense (356, activation='relu'),
            tf.keras.layers.Dense (1, activation='sigmoid')
        ]

model = tf.keras.models.Sequential(l)

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])		

''' # Again not my code

# PLOT LOSS AND ACCURACY
%matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

# Desired output. Charts with training and validation metrics. No crash :)

'''

# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Question.ipynb