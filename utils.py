import matplotlib.pyplot as plt
import seaborn as sns

def tensorboard_output(log_dir):

    print("Run the command line:\n" \
          "--> tensorboard --logdir={0} " \
          "\nThen open localhost:6006/ into your web browser".format(log_dir))