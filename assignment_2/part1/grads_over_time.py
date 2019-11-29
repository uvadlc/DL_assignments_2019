
import torch
from part1.dataset import PalindromeDataset
from torch.utils.data import DataLoader
import numpy as np
#from torch.autograd import Variable
import math

seq_length=5
batch_size=128

dataset_train = PalindromeDataset(seq_length + 1)
data_loader_train = DataLoader(dataset_train, batch_size, num_workers=1)

for step, (batch_inputs, batch_targets) in enumerate(data_loader_train):
    x, y = (batch_inputs_test, batch_targets_test)
    x_N_test, h = model.forward(x)
    print(x_N_test)

# seq_length=5
#
#
# #tried to load data here, but not really correct
# batch_size=128
# dataset_test = PalindromeDataset(seq_length + 1)
# data_loader_test= DataLoader(dataset_test, batch_size, num_workers=1)
# model=torch.load("RNN_seqlen_5_palin_model.pt")
# h_gradients = []
#
# enumerate(data_loader_test)

#
# left = [np.random.randint(0, 10) for _ in range(math.ceil(seq_length/2))]
# left = np.asarray(left, dtype=np.float32)
# right = np.flip(left, 0) if seq_length % 2 == 0 else np.flip(left[:-1], 0)
# full_palindrome= np.concatenate((left, right))
# palin=full_palindrome[0:-1], int(full_palindrome[-1])

(batch_inputs_test, batch_targets_test)=  palin
x, y = (batch_inputs_test, batch_targets_test)
#convert array to tensor

#x=Variable(torch.Tensor(x.values))
#says it needs matrices or tensors as input confusing
x_N_test, h = model.forward(x)

print(x_N_test)

loss = criterion(x_N_test, y_batch)
orig_test, predicted_test = torch.max(x_N_test.data, 1)

print(predicted_test)

# #
# #    print(x)
#
#     x_batch = x.clone().detach()  # dunno if i should detach
#     # y_batch= torch.tensor(y, dtype=torch.long)
#     y_batch = y.clone().detach()
#
#     # do one forward pass
#     x_N_test, h = model.forward(x=x_batch)
#
#     # calculate the loss
#     loss = criterion(x_N_test, y_batch)
#
#     # total_predictions = 0
#     # correct_predictions = 0.0
#
#     orig_test, predicted_test = torch.max(x_N_test.data, 1)
#
#     print(predicted_test)
#
#     # get performance of current testing step
#     total_predictions = predicted_test.size(0)
#     correct_predictions = predicted.eq(y_batch.data).sum().item()
#     accuracy_current_test = 100 * (correct_predictions / total_predictions)
#
#     loss_current_test = loss.item()
#
#
#
#
#
# #
# # grads = {}
# # def save_grad(name):
# #     def hook(grad):
# #         grads[name] = grad
# #     return hook
# #
# # def plot_grad_flow(model.parameters()):
# #     ave_grads = []
# #     layers = []
# #     for n, p in named_parameters:
# #         if(p.requires_grad) and ("bias" not in n):
# #             layers.append(n)
# #             ave_grads.append(p.grad.abs().mean())
# #     plt.plot(ave_grads, alpha=0.3, color="b")
# #     plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
# #     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
# #     plt.xlim(xmin=0, xmax=len(ave_grads))
# #     plt.xlabel("Layers")
# #     plt.ylabel("average gradient")
# #     plt.title("Gradient flow")
# #     plt.grid(True)
# #
# #
# # plot_grad_flow()
#
# # steps = range(1, len(h_gradients) + 1)
# # h_gradient_title = 'Hidden state gradient for each time step of ' + model_type
# # plt.plot(steps, h_gradients)
# # plt.title(h_gradient_title)
# # plt.ylabel('Hidden state Gradient')
# # # plt.ylim(0, 105)
# # plt.xlabel("Step")
# # plt.savefig(folder + model_type + '_h_gradients.png')
# # plt.show()
#
#
#




