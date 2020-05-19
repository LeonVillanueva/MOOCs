# https://www.youtube.com/watch?v=NlpS-DhayQA
# ImageDataGenerator Class
# directory contains subdirectory of classes,

# Learning Rate : https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
# https://gombru.github.io/2018/05/23/cross_entropy_loss/

# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%204%20-%20Notebook.ipynb
 
DESIRED_ACCURACY = 0.999

    class myCallback (tf.keras.callbacks.Callback):
        def on_epoch_end (self, epoch, logs={}):
            if (logs.get('acc')>DESIRED_ACCURACY):
                print ('\n{} ACCURACY REACHED'.format(DESIRED_ACCURACY))
                self.model.stop_training=True

    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    
    i, j, ch = 150, 150, 3
    
    l = [
            tf.keras.layers.Conv2D (16, 3, input_shape=(i, j, ch), activation='relu'),
            tf.keras.layers.MaxPooling2D (2, 2),
            tf.keras.layers.Conv2D (32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D (2, 2),
            tf.keras.layers.Conv2D (64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D (2, 2),
            tf.keras.layers.Flatten (),
            tf.keras.layers.Dense (128, activation='relu'),
            tf.keras.layers.Dense (1, activation='sigmoid')
        ]
    
    model = tf.keras.models.Sequential(l)

    from tensorflow.keras.optimizers import RMSprop

    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        '/tmp/h-or-s',           # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        class_mode='binary'      # Since we use binary_crossentropy loss, we need binary labels
        )
        
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(
              train_generator,
              steps_per_epoch=8,  
              epochs=15,
              verbose=1,
              callbacks=[callbacks]
    )