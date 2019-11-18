# Assignment 2: Recurrent Neural Networks

The second assignment will cover the topic of Recurrent Neural Networks, including backpropagation over time and LSTMs. All details can be found in the PDF document **assignment_2.pdf**.

Unlike the first assignment, there are no unittests this time. We will use PyTorch and its autograd function throughout the assignment.

To execute the code, the working directory is assumed to be the main directory containing `part1` and `part2`. During grading, we will run your code from this working directory. In case you also want to execute your code in the part-specific directories, you can add the following two lines in your training file:
```Python
import sys
sys.path.append("..")
```

### Prerequisites

You can use the same environment that you used for the first assignment. 

The first task can be mostly performed on your own computer (CPU), but especially for the second task, you will require a GPU to speed up your training. Hence, we suggest to run experiments on SURFSARA. 

## Task 1. RNNs versus LSTMs

For the first task, you will compare vanilla Recurrent Neural Networks (RNN) with Long-Short Term Networks (LSTM). You have to implement both network modules in the files `lstm.py` and `vanilla_rnn.py` from scratch (i.e. you are not allowed to use `nn.LSTM` or `nn.Linear`, but work with functionalities like `nn.Parameter`). The palindrome dataset generation is provided in `dataset.py` and can be used without any changes. 

The file `train.py` gives a initial structure for training your models. Make sure to integrate all (hyper-)parameters that are given for the `ArgumentParser`. Feel free to add more parameters if needed.

## Task 2. Text Generation

In the second task, you will use the built-in LSTM function, `nn.LSTM`, to generate text.

### Training Data

Make sure you download the books as plain text (.txt) file. Possible sources to get books from are:

1. Project Gutenberg, Dutch: https://www.gutenberg.org/browse/languages/nl
2. Project Gutenberg, English: https://www.gutenberg.org/browse/languages/en

Feel free to use other languages and/or other sources. Remember to include the datasets/books you used for training in your submission.

### Bonus questions

The assignment contains two bonus questions in which you can experiment further with your model. Note that they are not strictly necessary to get full points on this assignment (max. 100 points), and might require more effort compared to other questions for the same amount of points.

## Task 3. Graph Neural Networks

In the third task, you will have to answer a pen-and-paper questions about Graph Neural Networks. No implementation will be needed. Sources for explaining GNNs can be found in the assignment.

## Report

Similar to the first assignment, we expect you to write a small report on the study of recurrent neural networks, in which you answer all the questions in the assignment. Please, make your report to be self-contained without this README.md file. Include all results, used hyperparameters, etc. to reproduce your experiments. 

### Deliverables

Create zip archive with the following structure:

```
lastname_assignment_2.zip
│   report_lastname.pdf
│   part_1/
│      dataset.py
│      lstm.py
│      train.py
│      vanilla_rnn.py
|      grads_over_time.py
│   part_2/
│      dataset.py
│      model.py
│      train.py
|      assets/
```

Replace `lastname` with your last name. In case you have created any other python files to obtain results in the report, include those in the corresponding directory as well. Remember to replace the data in the assets folder with the ones you have used (i.e. put the txt files of the books you used in this folder). Given example datasets which were not used do not have to be included in the submission.

Finally, please submit your zip-file on Canvas.

The deadline for the assignment is the **27 November, 23:59**.

