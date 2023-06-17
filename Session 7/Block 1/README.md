# ERA-V1
Session 7 Assignemnt Block 1

## The requirements
The requirments of the project are in requirements.txt 

## Module structure
1. utils.py - contains the the utility functions
2. model.py - contains the module related classes and functions
3. S7 - Block1.ipynb - contains the driver code which calls the functions from utils and model
4. readme - file is there for other people to understand the over all flow of the project.

# Sample Data
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/7ed9da4e-eee0-4655-ad72-cde2219c72ac)

# Block 1
1. Target - To achive 99.4% accuracy with a bulky model
2. Result - The model is quite stable and achives 99.4% in 13th Epoch but with 16.5k param
3. Analysis - Although it is stable but it's bulky and dosen't make use of activation function

# Model summmary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #

            Conv2d-1           [-1, 16, 26, 26]             160
       BatchNorm2d-2           [-1, 16, 26, 26]              32
           Dropout-3           [-1, 16, 26, 26]               0
            Conv2d-4           [-1, 32, 24, 24]           4,640
       BatchNorm2d-5           [-1, 32, 24, 24]              64
           Dropout-6           [-1, 32, 24, 24]               0
            Conv2d-7            [-1, 8, 24, 24]             264
         MaxPool2d-8            [-1, 8, 12, 12]               0
            Conv2d-9           [-1, 16, 10, 10]           1,168
      BatchNorm2d-10           [-1, 16, 10, 10]              32
          Dropout-11           [-1, 16, 10, 10]               0
           Conv2d-12             [-1, 32, 8, 8]           4,640
      BatchNorm2d-13             [-1, 32, 8, 8]              64
          Dropout-14             [-1, 32, 8, 8]               0
           Conv2d-15             [-1, 16, 8, 8]             528
        MaxPool2d-16             [-1, 16, 4, 4]               0
           Conv2d-17             [-1, 32, 2, 2]           4,640
        AvgPool2d-18             [-1, 32, 1, 1]               0
           Linear-19                   [-1, 10]             330
----------------------------------------------------------------

Total params: 16,562<br>
Trainable params: 16,562<br>
Non-trainable params: 0

Input size (MB): 0.00<br>
Forward/backward pass size (MB): 0.81<br>
Params size (MB): 0.06<br>
Estimated Total Size (MB): 0.87

# Training logs
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/7d43fe6e-6ad2-4f43-855d-d7e8c4707282)


# Training and Testing plot
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/955d1041-060d-4725-b5a5-97382494798f)

