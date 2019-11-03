# Assignment 1

 You can find all necessary instructions in **assignment_1.pdf**.

 We provide to you simple unit tests that can check your implementation. However be aware that even if all tests are passed it still doesn't mean that your implementation is correct. You can find tests in **unittests.py**. 
 
 

 We also provide a Conda environment you can use to install the necessary Python packages. 
 In order to use it on SURFSARA, follow this instructions:
 
    - add the following lines in your ".bashrc":
        module load 2019
        module load CUDA/10.0.130
        module load Anaconda3/2018.12
    
    - logout and login again
    
    - copy the **environment.yml** file on SURFSARA (eg. in your home directory)
    
    - move to the folder containing the **environment.yml** file
    
    - run the following command once:
        conda env create -f environment.yml

    - add the following line at the beginning of your experiment script (.sh), before running your Python script:
        source activate dl
    
for further information about Anaconda/Miniconda:

https://docs.conda.io/projects/conda/en/latest/
https://docs.conda.io/en/latest/miniconda.html
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
