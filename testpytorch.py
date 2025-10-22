# test to make sure torch works
import torch
x = torch.rand(5, 3)
print(x)

# test if virtual environment (venv) exists
import sys
print(sys.prefix)