# import tensorflow as tf
# # print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])

# # Check available devices
# devices = tf.config.list_physical_devices()
# print("Available devices:", devices)

# # Check if a GPU is available
# gpu_devices = tf.config.list_physical_devices('GPU')
# if gpu_devices:
#     print("GPU is available")
# else:
#     print("GPU is not available")


# !python3 -m pip install tensorflow[and-cuda]
# # Verify the installation:


# % python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

import torch
print(torch.cuda.is_available())  # This should return True if CUDA is available
print(torch.cuda.device_count())  # This should return the number of available GPUs


import pytorch_lightning as pl

print(torch.__version__)
print(pl.__version__)

# https://pytorch.org/ go here to download better version of torch if needed


# Clear cache
torch.cuda.empty_cache()

# You can also reset memory using
torch.cuda.reset_peak_memory_stats()