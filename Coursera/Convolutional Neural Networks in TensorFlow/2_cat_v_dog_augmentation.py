# https://github.com/keras-team/keras-preprocessing
# PREPROCESSING : https://keras.io/api/preprocessing/image/
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%204%20-%20Lesson%202%20-%20Notebook%20(Cats%20v%20Dogs%20Augmentation).ipynb

try:
    parent = '/tmp/cats-v-dogs'
    os.mkdir (parent)
    
    sub = ['training', 'testing']
    for t in sub:
        os.mkdir ( os.path.join (parent, t) )
    
    cvd = ['cats','dogs']
    for t in sub:
        for s in cvd:
            os.mkdir ( os.path.join (os.path.join (parent, t), s))
except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    source_list = os.listdir (SOURCE)
    clean_source = [s for s in source_list if os.path.getsize ( os.path.join ( SOURCE, s ) ) > 0] # checks for ZERO
    
    train_list = random.sample ( clean_source, int ( len(clean_source)*SPLIT_SIZE ) )
    test_list = list(set(clean_source) - set(train_list))
    
    for i in train_list:
        copyfile ( os.path.join (SOURCE,i) , os.path.join (TRAINING,i) )
    for i in test_list:
        copyfile ( os.path.join (SOURCE,i) , os.path.join (TESTING,i) )
		
TRAINING_DIR = '/tmp/cats-v-dogs/training'
train_datagen = ImageDataGenerator(
                                        rescale=1/255,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        rotation_range=45,
                                        zoom_range=0.2,
                                        shear_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2
                                        )

# NOTE: YOU MUST USE A BATCH SIZE OF 10 (batch_size=10) FOR THE 
# TRAIN GENERATOR.
train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,           
        target_size=(150, 150),  
        class_mode='binary',
        batch_size=10
        )