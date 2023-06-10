# ERA-V1
Second git related assignment from session 6. The session 6 contains 2 parts. Part 1 deals with details of back propogation and it's literal implementation in a excel sheet for a very simple neural network. The part 2 challenges us to make a neural network that would achive 99.4% acccuracy on MNSIT dataset with some constraints on the model design.

## Excel and Back Propagation (Part 1)
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/f5270b8b-f96a-4236-af30-0937d935c00f)
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/91cffd02-9a84-4843-8ceb-62e941a7738e)
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/59d54f80-2bde-477c-925d-3a69d0b51f0b)
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/3372a4d2-5842-4f2f-a887-689fcfc66738)
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/19c8bd02-012d-42f2-a856-58a9bbf16cdc)
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/8f49f40c-d80e-4089-b446-d36de9ea245b)
![image](https://github.com/SudoWodo/ERA-V1/assets/82159869/6db9c538-11dc-404a-a37a-a523915d2f68)



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
