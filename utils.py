import matplotlib.pyplot as plt
import seaborn as sns

import os
import shutil

def tensorboard_output(log_dir):

    print("Run the command line:\n" \
          "--> tensorboard --logdir={0} " \
          "\nThen open localhost:6006/ into your web browser".format(log_dir))

def wipe_dir(log_dir):

    """delete a directory and replace"""

    try:
        shutil.rmtree(log_dir)
    except FileNotFoundError:
        pass

    os.mkdir(log_dir)