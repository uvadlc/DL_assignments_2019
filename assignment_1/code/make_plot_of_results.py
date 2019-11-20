folder = 'pytorch_results/'
acc_title = 'Accuracy of Pytorch MLP'
loss_title = 'Cross Entropy of Loss Pytorch MLP'
EVAL_FREQ_DEFAULT = 100



import matplotlib.pyplot as plt
import numpy as np

acc_test=np.load(folder+'acc_test.npy')
acc_train=np.load(folder+'acc_train.npy')
loss_test=np.load(folder+'loss_test.npy')
loss_train=np.load(folder+'loss_train.npy')


nr_evaluations = len(acc_test)

x_grid = np.linspace(1*EVAL_FREQ_DEFAULT,nr_evaluations*EVAL_FREQ_DEFAULT,nr_evaluations)


# summarize history for root mean_squared_error

plt.figure(1)
plt.plot(x_grid,loss_train)
plt.plot(x_grid,loss_test)
plt.title(loss_title)
plt.grid(True)
plt.ylabel('Loss')
plt.xlabel("Steps")
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(folder+'loss_curve.png')
plt.show()


# summarize history for accuracy
plt.figure(2)
plt.plot(x_grid,acc_train)
plt.plot(x_grid,acc_test)
plt.title(acc_title)
plt.grid(True)
plt.ylabel('Accuracy')
plt.xlabel("Steps")
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(folder+'acc_curve.png')
plt.show()


print('Acc test set')
print(acc_test[-1])
print('Acc train set')
print(acc_train[-1])
print('Loss test set')
print(loss_test[-1])
print('Loss train set')
print(loss_train[-1])
