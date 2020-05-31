# http://www.laurencemoroney.com/rock-paper-scissors-dataset/
# categorical cross entropy
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb

# https://www.kaggle.com/datamunge/sign-language-mnist

# CONVOLUTED that really should have just used pandas

q = f"{getcwd()}/../tmp2/sign_mnist_train.csv"
with open(q) as qq:
    
    q_data = csv.reader (qq)
    next (q_data)
    
    labels = []
    pics = []
            
    for row in q_data:
        label = row[0]
        data = row[1:]
        pic = np.array_split (np.array(data), 28)
        
        labels.append(label)
        pics.append(pic)

labels = np.array(labels).astype ('float')
pics = np.array(pics).astype ('float')

def get_data(filename):
    with open(filename) as training_file:
      # Your code starts here
        csvreader = csv.reader (training_file)
        next (csvreader)
        
        labels = []
        pics = []
            
        for row in csvreader:
            label = row[0]
            data = row[1:]
            pic = np.array_split (np.array(data), 28)
        
            labels.append(label)
            pics.append(pic)

        labels = np.array(labels).astype ('float')
        images = np.array(pics).astype ('float')
    
      # Your code ends here
    return images, labels
	
	
training_images = np.expand_dims (training_images, axis=3)
testing_images = np.expand_dims (testing_images, axis=3)

# (27455, 28, 28, 1)


i, j, ch = 28, 28, 1

l = [tf.keras.layers.Conv2D (16, 3, input_shape=(i, j, ch), activation='relu'),
     tf.keras.layers.MaxPooling2D (2, 2),
     tf.keras.layers.Conv2D (32, 3, activation='relu'),
     tf.keras.layers.MaxPooling2D (2, 2),
     tf.keras.layers.Flatten (),
     tf.keras.layers.Dense (256, activation='relu'),
     tf.keras.layers.Dense (128, activation='relu'),
     tf.keras.layers.Dense (3, activation='softmax')]

model = tf.keras.models.Sequential(l)

# https://keras.io/api/preprocessing/image/

# Define the model
# Use no more than 2 Conv2D and 2 MaxPooling2D

i, j, ch = 28, 28, 1

l = [tf.keras.layers.Conv2D (16, 3, input_shape=(i, j, ch), activation='relu'),
     tf.keras.layers.MaxPooling2D (2, 2),
     tf.keras.layers.Conv2D (32, 3, activation='relu'),
     tf.keras.layers.MaxPooling2D (2, 2),
     tf.keras.layers.Flatten (),
     tf.keras.layers.Dense (256, activation='relu'),
     tf.keras.layers.Dense (128, activation='relu'),
     tf.keras.layers.Dense (32, activation='softmax')]

model = tf.keras.models.Sequential(l)

# Compile Model. 
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the Model
history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=32), epochs=2, steps_per_epoch=len(training_images)/32, validation_data = validation_datagen.flow (testing_images, testing_labels))

model.evaluate(testing_images, testing_labels, verbose=0)

# https://www.kaggle.com/datamunge/sign-language-mnist