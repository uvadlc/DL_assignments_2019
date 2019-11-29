folder = 'results/'
model_type='LSTM'
acc_title = 'Accuracy of '+ model_type
loss_title = 'Cross Entropy of' + model_type
#EVAL_FREQ_DEFAULT = 100

import matplotlib.pyplot as plt
import numpy as np

acc_filename="LSTM_acc_test_palindrome_len"
loss_filename='LSTM_loss_test_palindrome_len'
palin_length_file_name="palins_tried"


def make_plot_over_steps_palin():
    path_to_acc_file=folder +acc_filename+".npy"
    path_to_loss_file=folder +loss_filename+".npy"
    path_to_steps_file= folder +palin_length_file_name +".npy"

    acc= np.load(path_to_acc_file)
    loss=np.load(path_to_loss_file)
    palin_lens=np.load(path_to_steps_file)

    print(loss)
    print(acc)
    print(palin_lens)

    acc_title = 'Accuracy of ' + model_type
    loss_title = 'Cross Entropy Loss of ' + model_type

    plt.plot(palin_lens, acc)
    plt.title(acc_title)
    plt.ylabel('Accuracy')
    #plt.ylim(0, 105)
    plt.xlabel("Training steps")
    plt.savefig(folder + model_type+'_acc_curve.png')
    plt.show()

    plt.plot(palin_lens, loss)
    plt.title(loss_title)
    plt.ylabel('Loss')
    # plt.ylim(0, 105)
    plt.xlabel("Training steps")
    plt.savefig(folder + model_type + '_loss_curve.png')
    plt.show()

make_plot_over_steps_palin()








#
#
#
# # acc_train=np.load(folder+'acc_train.npy')
# # loss_train=np.load(folder+'loss_train.npy')
#
# folder= ""
#
# filename_acc=folder + model_type + "_loss.npy" # loss_list)
# filename_loss=folder + model_type +  "_acc.npy" #acc_list)
#
# loss_palin=filename_loss
#
#
# #nr_evaluations = len(acc_test)
#
# #x_grid = np.linspace(1*EVAL_FREQ_DEFAULT,nr_evaluations*EVAL_FREQ_DEFAULT,nr_evaluations)
# x_grid=np.linspace(10,10)
# # summarize history for root mean_squared_error
#
# plt.figure(1)
# plt.plot(x_grid,loss_palin)
# #plt.plot(x_grid,loss_test)
# plt.title(loss_title)
# plt.grid(True)
# plt.ylabel('Loss')
# plt.xlabel("Palindrome Length")
# #plt.legend(['train', 'test'], loc='upper left')
# plt.savefig(folder+'loss_curve.png')
# plt.show()
#
# #
# # # summarize history for accuracy
# # plt.figure(2)
# # plt.plot(x_grid,acc_train)
# # plt.plot(x_grid,acc_test)
# # plt.title(acc_title)
# # plt.grid(True)
# # plt.ylabel('Accuracy')
# # plt.xlabel("Steps")
# # plt.legend(['train', 'test'], loc='upper left')
# # plt.savefig(folder+'acc_curve.png')
# # plt.show()
#
#
# # print('Acc test set')
# # print(acc_test[-1])
# # print('Acc train set')
# # print(acc_train[-1])
# # print('Loss test set')
# # print(loss_test[-1])
# # print('Loss train set')
# # print(loss_train[-1])
