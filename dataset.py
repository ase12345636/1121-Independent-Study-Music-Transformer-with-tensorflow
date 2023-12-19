from midi_processor import midi_Tokenization
import os
import numpy as np

train1=[]
train2=[]
for i in range (1,2):
    path=os.path.dirname(os.path.abspath(__file__))
    path+="/Dataset/train/train ("+ str(i) +").midi"
    encoded0, encoded1 = midi_Tokenization(path)
    train1.append(encoded0)
    train2.append(encoded1)
print("Finsh training dataset")

vaild1=[]
vaild2=[]
for i in range (1,2):
    path=os.path.dirname(os.path.abspath(__file__))
    path+="/Dataset/vaild/vaild ("+ str(i) +").midi"
    encoded0, encoded1 = midi_Tokenization(path)
    vaild1.append(encoded0)
    vaild2.append(encoded1)


print("Finsh vailding dataset")


'''
for i in range (1,138):
    path=os.path.dirname(os.path.abspath(__file__))
    path+="/Dataset/vaild/vaild ("+ str(i) +").midi"
    encoded = midi_Tokenization(path)
    resource,target = np.array_split (encoded,2)

for i in range (1,178):
    path=os.path.dirname(os.path.abspath(__file__))
    path+="/Dataset/train/test ("+ str(i) +").midi"
    encoded = midi_Tokenization(path)
    print(encoded)
    resource,target = np.array_split (encoded,2)
'''