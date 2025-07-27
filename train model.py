import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# ✅ Enable Mixed Precision & XLA Compilation for faster training
mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

# ✅ GPU Optimization
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# --- CONFIG ---
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 2
MODEL_PATH = "shoplifting_twopath_model_fast.h5"

# ✅ Load and Recompile the Saved Model
print("\n🔄 Loading model...")
model = tf.keras.models.load_model("shoplifting_twopath_model.h5")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("✅ Model recompiled successfully!")

# ✅ Checkpointing
checkpoint_callback = ModelCheckpoint(
    MODEL_PATH,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

# --- IMAGE LOADING FUNCTION ---
def load_image(rgb_path, flow_path, label):
    """
    Safely loads and preprocesses RGB and flow images.
    """
    # ✅ Decode TensorFlow Strings
    rgb_path = rgb_path.numpy().decode("utf-8")
    flow_path = flow_path.numpy().decode("utf-8")

    # ✅ Read images using OpenCV
    rgb_img = cv2.imread(rgb_path)
    flow_img = cv2.imread(flow_path)

    if rgb_img is None or flow_img is None:
        print(f"⚠️ Missing image: {rgb_path} or {flow_path}")
        rgb_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        flow_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
    else:
        # ✅ Resize and normalize
        rgb_img = cv2.resize(rgb_img, IMG_SIZE) / 255.0
        flow_img = cv2.resize(flow_img, IMG_SIZE) / 255.0

    return rgb_img, flow_img, label


def process_paths(rgb_path, flow_path, label):
    """
    Applies the loading function and ensures shapes.
    """
    rgb, flow, label = tf.py_function(
        func=load_image,
        inp=[rgb_path, flow_path, label],
        Tout=(tf.float32, tf.float32, tf.int64)
    )

    # ✅ Ensure Fixed Shapes After py_function
    rgb = tf.ensure_shape(rgb, (*IMG_SIZE, 3))
    flow = tf.ensure_shape(flow, (*IMG_SIZE, 3))
    label = tf.ensure_shape(label, ())

    return (rgb, flow), label


def create_dataset(rgb_files, flow_files, folder_rgb, folder_flow, batch_size):
    """
    Creates a TensorFlow dataset pipeline.
    """
    # ✅ Use TensorFlow-native string decoding
    rgb_paths = [os.path.join(folder_rgb, f) for f in rgb_files]
    flow_paths = [os.path.join(folder_flow, f) for f in flow_files]
    labels = np.array([1 if "shoplifting" in f else 0 for f in rgb_files], dtype=np.int64)

    # ✅ Create Dataset
    dataset = tf.data.Dataset.from_tensor_slices((rgb_paths, flow_paths, labels))

    # ✅ Decode EagerTensors safely using `tf.strings`
    def decode_fn(rgb_path, flow_path, label):
        rgb_path = tf.strings.join([rgb_path])
        flow_path = tf.strings.join([flow_path])
        return rgb_path, flow_path, label

    dataset = dataset.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # ✅ Efficient Image Loading
    dataset = dataset.map(
        lambda x, y, z: process_paths(x, y, z),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # ✅ Optimize the pipeline
    dataset = dataset.cache().shuffle(len(rgb_files)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"✅ Dataset created with {len(rgb_files) // batch_size} batches")

    return dataset


# ✅ Load and Split Data
rgb_folder, flow_folder = "data/frames/rgb", "data/frames/flow"
rgb_filenames = sorted([f for f in os.listdir(rgb_folder) if f.endswith(('.jpg', '.png'))])
flow_filenames = sorted([f for f in os.listdir(flow_folder) if f.endswith(('.jpg', '.png'))])

# ✅ Split into training and testing sets
train_rgb, test_rgb, train_flow, test_flow = train_test_split(rgb_filenames, flow_filenames, test_size=0.2)

# ✅ Create datasets
train_dataset = create_dataset(train_rgb, train_flow, rgb_folder, flow_folder, BATCH_SIZE)
test_dataset = create_dataset(test_rgb, test_flow, rgb_folder, flow_folder, BATCH_SIZE)

# ✅ Training
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback]
)
