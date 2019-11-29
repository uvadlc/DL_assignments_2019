
import numpy as np
from matplotlib import pyplot as plt

#read in the results

#curent_folder=os.getcwd()
folder= "./part2/final_grim/"


model_type="LSTM"
acc_filename="LSTM_acc30"
loss_filename= "LSTM_loss30"

acc_filename= "LSTM_loss_temp_0_seq_30"
loss_filename="LSTM_acc_temp_0_seq_30"

def make_plot_over_steps():
    path_to_acc_file=folder +acc_filename+".npy"
    path_to_loss_file=folder +loss_filename+".npy"

    acc= np.load(path_to_acc_file)
    loss=np.load(path_to_loss_file)

    acc_title = 'Accuracy of ' + model_type
    loss_title = 'Cross Entropy Loss of ' + model_type

    plt.plot( acc)
    plt.title(acc_title)
    plt.ylabel('Accuracy')
    #plt.ylim(0, 105)
    plt.xlabel("Training steps")
    plt.savefig(folder + model_type+'_acc_curve.png')
    plt.show()

    plt.plot(loss)
    plt.title(loss_title)
    plt.ylabel('Loss')
    # plt.ylim(0, 105)
    plt.xlabel("Training steps")
    plt.savefig(folder + model_type + '_loss_curve.png')
    plt.show()



make_plot_over_steps()

#range(1:len(x_axis)+1),



