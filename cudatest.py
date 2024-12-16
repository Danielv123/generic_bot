import cv2
import torch
print(cv2.getBuildInformation())

print("CV2 GPU support:")
count = cv2.cuda.getCudaEnabledDeviceCount()
print(count)

print("Pytorch cuda support:")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
    print(torch.cuda.get_device_properties(i))
