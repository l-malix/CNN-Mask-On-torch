import cv2
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision import transforms
from torchvision.transforms.transforms import ToPILImage


results={0:'mask',1:'no mask'}
GR_dict={1:(0,0,255),0:(0,255,0)}

class maskOnNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 100, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(100, 100, 3)
        self.fc1 = nn.Linear(100*28*28, 50)
        self.fc2 = nn.Linear(50, 2)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 100*28*28)
        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = maskOnNet()
model.load_state_dict(torch.load('./LbesLmask.pth'))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((120,120)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

rect_size = 4

while True:

    _, frame = cap.read()

    
    frame=cv2.flip(frame,1,1)
    rerect_size = cv2.resize(frame, (frame.shape[1] // rect_size, frame.shape[0] // rect_size))
    faces = face_cascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]
        
        face_img = frame[y:y+h, x:x+w]
        face_img = face_img[:,:,::-1]
        face_img = trans(face_img)

        result = model(face_img.view(1,3,120,120))
        
        _, pred = torch.max(result, 1)

        print(pred)
      
        cv2.rectangle(frame,(x,y),(x+w,y+h),GR_dict[pred.item()],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),GR_dict[pred.item()],-1)
        cv2.putText(frame, results[pred.item()], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
    cv2.imshow('LIVE',   frame)
    key = cv2.waitKey(10)
    if key == 27: 
        break
cap.release()
cv2.destroyAllWindows()