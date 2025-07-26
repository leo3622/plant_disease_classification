import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# Set number of images to use (Adjust as needed)
NUM_TRAIN_SAMPLES = 12000   # Change this to the number of images you want for training
NUM_VAL_SAMPLES = 2500     # Change this for validation
NUM_TEST_SAMPLES = 2500    # Change this for testing

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_PATH = "/kaggle/input/plantvillage-dataset/color"

# On local machine, set the DATA_PATH to the path where your dataset is stored after downloading it from Kaggle.
# For example:
    #DATA_PATH = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset/color")

def load_dataset():
    train_ds_full = image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2,  # 80% train, 20% temp
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    temp_ds = image_dataset_from_directory(
        DATA_PATH,
        validation_split=0.2,  # 20% temp (validation + test)
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Get class names
    class_names = train_ds_full.class_names
    num_classes = len(class_names)

    # Further split temp_ds into validation and test
    val_size = tf.data.experimental.cardinality(temp_ds).numpy() // 2
    val_ds_full = temp_ds.take(val_size)  
    test_ds_full = temp_ds.skip(val_size)  

    train_ds = train_ds_full.take(NUM_TRAIN_SAMPLES // BATCH_SIZE)
    val_ds = val_ds_full.take(NUM_VAL_SAMPLES // BATCH_SIZE)
    test_ds = test_ds_full.take(NUM_TEST_SAMPLES // BATCH_SIZE)

    def preprocess_both(image, label):

        teacher_image = resnet_preprocess(image)
        student_image = mobilenet_preprocess(image)
        
        return (teacher_image, student_image), label

    train_ds = train_ds.map(preprocess_both).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_both).cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_both).cache().prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names, num_classes