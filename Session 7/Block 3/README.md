# ERA-V1
Session 7 Assignemnt Block 3

## The requirements
The requirments of the project are in requirements.txt 

## Module structure
1. utils.py - contains the the utility functions
2. model.py - contains the module related classes and functions
3. S7 - Block3.ipynb - contains the driver code which calls the functions from utils and model
4. readme - file is there for other people to understand the over all flow of the project.

# Sample Data
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/7ed9da4e-eee0-4655-ad72-cde2219c72ac)

# Block 3
1. Target - To achive 99.4 before 15th epoch i.e on or before 14th epoch.
2. Result - We achived 99.41 on 13th epoch and maitained 99.47 on 14th epoch.
3. Analysis - increasing learing rate from 0.01 to 0.02 and decrease momentum from 0.9 to 0.8 helped but slighty unstable 

# Model summmary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #

            Conv2d-1           [-1, 32, 26, 26]             288
              ReLU-2           [-1, 32, 26, 26]               0
       BatchNorm2d-3           [-1, 32, 26, 26]              64
         Dropout2d-4           [-1, 32, 26, 26]               0
            Conv2d-5            [-1, 8, 26, 26]             256
              ReLU-6            [-1, 8, 26, 26]               0
       BatchNorm2d-7            [-1, 8, 26, 26]              16
         Dropout2d-8            [-1, 8, 26, 26]               0
            Conv2d-9           [-1, 10, 24, 24]             720
             ReLU-10           [-1, 10, 24, 24]               0
      BatchNorm2d-11           [-1, 10, 24, 24]              20
        Dropout2d-12           [-1, 10, 24, 24]               0
        MaxPool2d-13           [-1, 10, 12, 12]               0
           Conv2d-14           [-1, 16, 12, 12]           1,440
             ReLU-15           [-1, 16, 12, 12]               0
      BatchNorm2d-16           [-1, 16, 12, 12]              32
        Dropout2d-17           [-1, 16, 12, 12]               0
           Conv2d-18           [-1, 10, 12, 12]             160
             ReLU-19           [-1, 10, 12, 12]               0
      BatchNorm2d-20           [-1, 10, 12, 12]              20
        Dropout2d-21           [-1, 10, 12, 12]               0
           Conv2d-22           [-1, 10, 10, 10]             900
             ReLU-23           [-1, 10, 10, 10]               0
      BatchNorm2d-24           [-1, 10, 10, 10]              20
        Dropout2d-25           [-1, 10, 10, 10]               0
           Conv2d-26             [-1, 16, 8, 8]           1,440
             ReLU-27             [-1, 16, 8, 8]               0
      BatchNorm2d-28             [-1, 16, 8, 8]              32
        Dropout2d-29             [-1, 16, 8, 8]               0
           Conv2d-30             [-1, 16, 6, 6]           2,304
             ReLU-31             [-1, 16, 6, 6]               0
      BatchNorm2d-32             [-1, 16, 6, 6]              32
        Dropout2d-33             [-1, 16, 6, 6]               0
        AvgPool2d-34             [-1, 16, 1, 1]               0
           Conv2d-35             [-1, 10, 1, 1]             160

Total params: 7,904 <br>
Trainable params: 7,904 <br>
Non-trainable params: 0 <br>

Input size (MB): 0.00 <br>
Forward/backward pass size (MB): 1.21 <br>
Params size (MB): 0.03 <br>
Estimated Total Size (MB): 1.24 <br>

# Training logs
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/a371fcb1-acd8-40cf-91ed-c429d2830924)


# Training and Testing plot
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/d851d088-143e-49f3-9c5f-ad11c6eef235)
