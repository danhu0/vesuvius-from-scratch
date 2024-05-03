import tensorflow as tf

# Check available devices
devices = tf.config.list_physical_devices()
print("Available devices:", devices)

# Check if a GPU is available
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("GPU is available")
else:
    print("GPU is not available")

# !python3 -m pip install tensorflow[and-cuda]
# # Verify the installation:


# % python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"