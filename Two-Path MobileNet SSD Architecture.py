import tensorflow as tf

# --- LOAD THE SAVED MODEL ---
model = tf.keras.models.load_model("shoplifting_twopath_model.h5")
print("\n✅ Model loaded successfully!")

# --- DATA GENERATOR ---
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# --- LOAD RGB & FLOW FRAMES ---
train_rgb = datagen.flow_from_directory("data/frames/rgb", target_size=(128, 128), batch_size=32, class_mode='binary',
                                        seed=42)
train_flow = datagen.flow_from_directory("data/frames/flow", target_size=(128, 128), batch_size=32, class_mode='binary',
                                         seed=42)


# --- ZIPPING RGB AND FLOW GENERATORS ---
def combined_generator(gen_rgb, gen_flow):
    for (rgb_batch, rgb_labels), (flow_batch, flow_labels) in zip(gen_rgb, gen_flow):
        # Ensure labels match
        assert (rgb_labels == flow_labels).all(), "Labels do not match!"

        # Yield combined inputs and labels
        yield [rgb_batch, flow_batch], rgb_labels


# --- COMBINE GENERATORS ---
combined_gen = combined_generator(train_rgb, train_flow)

# --- TRAIN THE MODEL ---
history = model.fit(
    combined_gen,
    steps_per_epoch=min(len(train_rgb), len(train_flow)),
    epochs=20
)

# --- SAVE THE UPDATED MODEL ---
model.save("shoplifting_twopath_model_updated.h5")
print("\n✅ Model saved successfully as 'shoplifting_twopath_model_updated.h5'!")
