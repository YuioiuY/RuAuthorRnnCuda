import tensorflow as tf

# GPU acceleration
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU for training!")
else:
    print("No GPU detected, using CPU.")