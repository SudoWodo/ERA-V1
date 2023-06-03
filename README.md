# ERA-V1
First git related assignment from session 5

## The requirements
The requirments of the project are in requirements.txt 

## Module structure
1. utils.py - contains the the utility functions
2. model.py - contains the module related classes and functions
3. S5.ipynb - contains the driver code which calls the functions from utils and model
4. readme - file is there for other people to understand the over all flow of the project.

# Sample Data
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/7ed9da4e-eee0-4655-ad72-cde2219c72ac)

# Model summmary

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #

            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
----------------------------------------------------------------
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0

Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.2
Estimated Total Size (MB): 2.94

# Training and Testing plot 2 epoch
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/a5197b7d-e5a0-4d3f-af43-1bc97f7b2ec1)
