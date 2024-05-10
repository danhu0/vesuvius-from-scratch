# vesuvius-from-scratch

Team Members: Daniel Hu (danhu0), Eitan Zemel (EitanZe), Jacob Stifelman (JMS100), Jason Tapper (jasontapper)

Run the check_gpu.py script to verify that you have a GPU that will properly run our TimeSformer model in a reasonable amount of time. This will require about 10-20 GB of GPU RAM in addition to ~15 GB of System RAM. The dataset is about 30 GB of data. If you want to run the original Kaggle tutorial notebook, you will need the same amount of disk storage, and a little under 4gb of GPU RAM.

If you are running the model locally on a machine with a GPU but it is not being reflected in the check_gpu script, reinstall a version of pytorch with CUDA support. Find the appropriate pip/conda command here: https://pytorch.org/

Our original code is adapted from the Kaggle Notebook here (https://www.kaggle.com/code/jpposma/vesuvius-challenge-ink-detection-tutorial) as well as the download for the dataset we used. In order to download the dataset, you will have to sign into Kaggle and agree to the terms of the Vesuvius Challenge there. Place the train and test folders unchanged in the root directory of your repo. For a version of our final notebook that we adapted to work with Google Colab, place the dataset directly into your Google Drive and copy the following notebook: https://drive.google.com/file/d/1MwZlKFW4Y8zZdVVWfM_rWaze2fGn1ETz/view?usp=sharing

For manually downloading other segments/fragments to train on, we used scripts from https://github.com/JamesDarby345/VesuviusDataDownload. To run the scripts from here, you will need a machine/virtual environment with rclone. Running this locally takes a lot a lot of space, so it's best done on a department machine. To run on a department machine, you will need to make a venv. This can be done with the command ```python3 -m venv name-of-venv```. Source your venv and then cd into the root project dir. Download the requirements via the requirements script. This will dump most of the required dependencies into the new venv, but you will need to get rclone manually.

To do this, go to the Debian Linux rclone link from the website and manually download it. Extract the contents of the folder (specifically extract the data.tar.gz file) and then extract the tarball of this file using tar -xzf data.tar.gz, which gives you the binary executable. Drop this into the bin of your venv and check rclone works with ```rclone --verison```.

Known Bugs:
The dimensions are a little bit off for the masking/buffer edge due to weird behavior with patching and requirements for divisibility, so if the model tries randomly to train on an edge pixel, it will instead train on a pixel in the middle of the fragment. There is not very much information on the edges of the fragment, so it should not affect the performance significantly
