import torch
from torch import nn
from PIL import Image
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import cv2 as cv2
import torchvision.transforms as transforms
import matplotlib
import math
import torch.optim as optim
import av
import time
import threading
matplotlib.use('Agg')

r = 2
q = 0.75
p = 0.75
r1 = 0.5
r2 = 0.25

Epoch = 1
learning_rate = 0.0001

filepath_train = "small.mp4"
filepath_gt = "sunny.mp4"

#check if cuda is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")

def get_videodetails(video_path):
    capture = cv2.VideoCapture(video_path)
    framecount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return framecount, width, height, fps

def video_to_tensor(video_path,frame_number):
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    out, frame = capture.read()

    # Convert the frame to a pytorch tensor and split it into the three channels
    #make it channel,width,height
    
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    
    frame_r = frame[0].unsqueeze(0)
    frame_g = frame[1].unsqueeze(0)
    frame_b = frame[2].unsqueeze(0)
    if not out:
        print("Error reading frame")
    capture.release()
    return frame_r,frame_g,frame_b

#tenso = video_to_tensor("plane2.mp4",2000)
#save_image(tenso,"E:/test/frame.jpg")

tmp1 = get_videodetails(filepath_train)
tmp2 = get_videodetails(filepath_gt)

videoFrameCount_train = tmp1[0]
videoWidth_train = tmp1[1]
videoHeight_train = tmp1[2]
videoFPS_train = tmp1[3]

videoFrameCount_gt = tmp2[0]
videoWidth_gt = tmp2[1]
videoHeight_gt = tmp2[2]
videoFPS_gt = tmp2[3]

torch.autograd.set_detect_anomaly(True)
    
class LERN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.hst = torch.rand(r**4,round(videoHeight_train/r),round(videoWidth_train/r)).to(device)
        
        self.pixel_unshuffle = nn.PixelUnshuffle(r)
        self.pixel_shuffle = nn.PixelShuffle(r)
        self.pixel_shuffle2 = nn.PixelShuffle(r**2)
        self.conv1 = nn.Conv2d(round(r**2),round(32*q),kernel_size=3,stride=1,padding=1)
        #self.concatinate = nn.Concatinate()

        #hst-1
        self.conv2 = nn.Conv2d(round(r**4),round(32*(1-q)),kernel_size=3,stride=1,padding=1)
        #

        #sra 32
        self.leaky_relu = nn.LeakyReLU(0.25)
        #3x3dsconv&relu depthwise separable convolution 
        self.conv3 = nn.Conv2d(32,round(32*r1),kernel_size=3,stride=1,padding=1,groups=round(r**2))
        #channel shuffle
        self.channel_shuffle = nn.ChannelShuffle(r)

        self.conv4 = nn.Conv2d(round(32*r1),32,kernel_size=3,stride=1,padding=1,groups=round(r**2))

        #attention block
        #self.globalaverage = nn.GlobalAveragePool2d()
        self.conv5 = nn.Conv2d(32,round(32*r2),kernel_size=3,stride=1,padding=1)

        self.conv6 = nn.Conv2d(round(32*r2),32,kernel_size=3,stride=1,padding=1)
        self.sigmoid = nn.Sigmoid()
        #
        self.conv7 = nn.Conv2d(32,round(16*p),kernel_size=3,stride=1,padding=1,groups=round(r**2))

        #lrt+1
        self.conv8 = nn.Conv2d(round(r**2),round(16*(1-p)),kernel_size=3,stride=1,padding=1)

        #sra 16
        self.conv9 = nn.Conv2d(16,round(16*r1),kernel_size=3,stride=1,padding=1,groups=round(r**2))
        self.conv10 = nn.Conv2d(round(16*r1),16,kernel_size=3,stride=1,padding=1,groups=round(r**2))
        self.conv11 = nn.Conv2d(16,round(16*r2),kernel_size=3,stride=1,padding=1)
        self.conv12 = nn.Conv2d(round(16*r2),16,kernel_size=3,stride=1,padding=1)
        #

        self.conv13 = nn.Conv2d(16,8,kernel_size=3,stride=1,padding=1,groups=round(r**2))
        self.conv14 = nn.Conv2d(8,round(r**4),kernel_size=3,stride=1,padding=1,groups=round(r**2))
        #self.conv13 = nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1,groups=round(r**2))
        #self.conv14 = nn.Conv2d(32,round(r**4),kernel_size=3,stride=1,padding=1,groups=round(r**2))

        #nets
        self.conv15 = nn.Conv2d(1,round(r**2),kernel_size=3,stride=1,padding=1)

        
    def init_hidden(self):
        return torch.rand(r**4,round(videoHeight_train/r),round(videoWidth_train/r)).to(device)    
    def forward(self, x,x1,htt):        
        #hstt = self.hst
        lrt = x
        lrt1 = x1
        hstt = htt
        #top of neth
        lrt = self.pixel_unshuffle(lrt)
        lrt = self.conv1(lrt)
        
        hstt = self.conv2(hstt)
        lrt = torch.cat((lrt,hstt),0)
        #sra part 32 
        lrrt = lrt
        lrt = self.leaky_relu(lrt)
        lrt = self.leaky_relu(self.conv3(lrt))
        #lrt = self.channel_shuffle(lrt)
        lrt = self.conv4(lrt)
        #lrt = self.channel_shuffle(lrt)
        lrrrt = lrt
        lrt = self.conv5(lrt)
        lrt = self.leaky_relu(lrt)
        lrt = self.conv6(lrt)
        lrt = self.sigmoid(lrt)
        lrt = lrrrt*lrt
        lrt = lrrt+lrt
        #

        lrt = self.conv7(lrt)

        #lrt+1
        lrt1 = self.pixel_unshuffle(lrt1)
        lrt1 = self.conv8(lrt1)

        lrt = torch.cat((lrt,lrt1),0)

        #sra part 16
        lrrt = lrt
        lrt = self.leaky_relu(lrt)
        lrt = self.leaky_relu(self.conv9(lrt))
        #lrt = self.channel_shuffle(lrt)
        lrt = self.conv10(lrt)
        #lrt = self.channel_shuffle(lrt)
        lrrrt = lrt
        lrt = self.conv11(lrt)
        lrt = self.leaky_relu(lrt)
        lrt = self.conv12(lrt)
        lrt = self.sigmoid(lrt)
        lrt = lrrrt*lrt
        lrt = lrrt+lrt
        #
        lrt = self.conv13(lrt)
        lrt = self.conv14(lrt)
        lrt = self.leaky_relu(lrt)
        hstt = lrt

        #nets
        x = self.conv15(x)
        x = self.pixel_shuffle(x)
        lrt = self.pixel_shuffle2(lrt)
        x = x+lrt

        #self.hst = hstt

        return x,hstt
#prepare loader
#input = input.to(device)
start = time.time()



def trainingRED():
    net = LERN()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    net.to(device)
    hidden = net.init_hidden().to(device)
    for i in range(Epoch): 
        print("Epoch: "+ str(i))   
        
        for x in range(videoFrameCount_train-1):
            #red
            red_train,green_train,blue_train = video_to_tensor(filepath_train,x)
            red_train1,green_train1,blue_train1 = video_to_tensor(filepath_train,x+1)
            red_gt,green_gt,blue_gt = video_to_tensor(filepath_gt,x)
            #set gradients to none
            optimizer.zero_grad()
            output,hidden = net(red_train.to(device),red_train1.to(device),hidden.detach())
            loss = criterion(output, red_gt.to(device))
            loss.backward()
            optimizer.step()
            #print(" Percentage done: "+str(x/videoFrameCount_train*100)+"            Time remaining:" + str(round((videoFrameCount_train-x)*(time.time()-start)/(x+1)/60)) + " minutes")




        
    print("done thread Red")
    torch.save(net.state_dict(), "modelRed.pth")    




def trainingGREEN():
    net = LERN()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    net.to(device)
    hidden = net.init_hidden().to(device)
    for i in range(Epoch): 
        print("Epoch: "+ str(i))   
        
        for x in range(videoFrameCount_train-1):
            #red
            red_train,green_train,blue_train = video_to_tensor(filepath_train,x)
            red_train1,green_train1,blue_train1 = video_to_tensor(filepath_train,x+1)
            red_gt,green_gt,blue_gt = video_to_tensor(filepath_gt,x)
            optimizer.zero_grad()
            output,hidden = net(green_train.to(device),green_train1.to(device),hidden.detach())
            loss = criterion(output, green_gt.to(device))
            loss.backward()
            optimizer.step()

            #print(" Percentage done: "+str(x/videoFrameCount_train*100)+"            Time remaining:" + str(round((videoFrameCount_train-x)*(time.time()-start)/(x+1)/60)) + " minutes")





    print("done thread Green")
    torch.save(net.state_dict(), "modelGreen.pth")    

    

def trainingBLUE():
    net = LERN()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    net.to(device)
    hidden = net.init_hidden().to(device)
    for i in range(Epoch): 
        print("Epoch: "+ str(i))   
        
        for x in range(videoFrameCount_train-1):
            #red
            red_train,green_train,blue_train = video_to_tensor(filepath_train,x)
            red_train1,green_train1,blue_train1 = video_to_tensor(filepath_train,x+1)
            red_gt,green_gt,blue_gt = video_to_tensor(filepath_gt,x)
            optimizer.zero_grad()
            output,hidden = net(blue_train.to(device),blue_train1.to(device),hidden.detach())
            loss = criterion(output, blue_gt.to(device))
            loss.backward()
            optimizer.step()
            print(" Percentage done: "+str(x/videoFrameCount_train*100)+"            Time remaining:" + str(round((videoFrameCount_train-x)*(time.time()-start)/(x+1)/60)) + " minutes"     + "   Error: " + str(loss.item()))


        #calcutate psnr


        #learning_rate = learning_rate /2
    print("done thread Blue")
    torch.save(net.state_dict(), "modelBlue.pth")    

#forwards pass and save video to file
def testing():
    device = torch.device("cpu")
    net_red = LERN()
    net_red.load_state_dict(torch.load("modelRed.pth"))
    net_red.to(device)
    hidden_red = net_red.init_hidden().to(device)
    net_green = LERN()
    net_green.load_state_dict(torch.load("modelGreen.pth"))
    net_green.to(device)
    hidden_green = net_green.init_hidden().to(device)
    net_blue = LERN()
    net_blue.load_state_dict(torch.load("modelBlue.pth"))
    net_blue.to(device)
    hidden_blue = net_blue.init_hidden().to(device)
    
    out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), videoFPS_train, (videoWidth_train*r,videoHeight_train*r))
    for x in range(videoFrameCount_train-1):

        red_train,green_train,blue_train = video_to_tensor(filepath_train,x)
        red_train1,green_train1,blue_train1 = video_to_tensor(filepath_train,x+1)
        
        output_red,hidden_red = net_red(red_train.to(device),red_train1.to(device),hidden_red.detach())
        output_green,hidden_green = net_green(green_train.to(device),green_train1.to(device),hidden_green.detach())
        output_blue,hidden_blue = net_blue(blue_train.to(device),blue_train1.to(device),hidden_blue.detach())
        
        output_red = output_red.detach().numpy()
        output_green = output_green.detach().numpy()
        output_blue = output_blue.detach().numpy()
        #save as mp4 video using cv2
        output = np.zeros((videoHeight_train*r,videoWidth_train*r,3), np.uint8)
        output[:,:,0] = output_red
        output[:,:,1] = output_green
        output[:,:,2] = output_blue
        out.write(output)
        #display video
        cv2.imshow('frame',output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        print(" Percentage done: "+str(x/videoFrameCount_train*100)+"            Time remaining:" + str(round((videoFrameCount_train-x)*(time.time()-start)/(x+1)/60)) + " minutes")
        

#run thread for each colour
#t1 = threading.Thread(target=trainingRED)
#t2 = threading.Thread(target=trainingGREEN)
#t3 = threading.Thread(target=trainingBLUE)

#t1.start()
#t2.start()
#t3.start()

#t1.join()
#t2.join()
#t3.join()

#thread for testing
t4 = threading.Thread(target=testing)
t4.start()
t4.join()
print("done")

