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
learning_rate = 0.001
batch = 24

filepath_train = "E:/test(1).mp4"
filepath_gt = "E:/big.mp4"

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

videoFrameCount_train,videoWidth_train,videoHeight_train,videoFPS_train = get_videodetails(filepath_train)
videoFrameCount_gt,videoWidth_gt,videoHeight_gt,videoFPS_gt = get_videodetails(filepath_gt)
batch_train = math.ceil(videoFrameCount_train/batch)
batch_gt = math.ceil(videoFrameCount_gt/batch)
def video_to_tensor(video_path,batch_number):
    capture = cv2.VideoCapture(video_path)
    framecount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #batch_count = math.ceil(framecount/batch)
    frame_r_list = []
    frame_g_list = []
    frame_b_list = []
    for i in range(batch):
        frame_index = batch * batch_number + i
        if frame_index >= framecount:
            print("Frame end reached!!!")
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES,frame_index)

        out, frame = capture.read()
        frame = torch.from_numpy(frame).permute(2, 0, 1).float().to(device)
    
        frame_r = frame[0].unsqueeze(0)
        frame_g = frame[1].unsqueeze(0)
        frame_b = frame[2].unsqueeze(0)
        if not out:
            print("Error reading frame")
            #Add save of models just in case of an error mid training
        frame_r_list.append(frame_r)
        frame_g_list.append(frame_g)
        frame_b_list.append(frame_b)
    frame_r = torch.cat(frame_r_list, dim=0).to(device)
    frame_g = torch.cat(frame_g_list, dim=0).to(device)
    frame_b = torch.cat(frame_b_list, dim=0).to(device)
    
    capture.release()
    return frame_r,frame_g,frame_b

#tenso = video_to_tensor("plane2.mp4",2000)
#save_image(tenso,"E:/test/frame.jpg")



#torch.autograd.set_detect_anomaly(True)
    
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

        #self.conv13 = nn.Conv2d(16,8,kernel_size=3,stride=1,padding=1,groups=round(r**2))
        #self.conv14 = nn.Conv2d(8,round(r**4),kernel_size=3,stride=1,padding=1,groups=round(r**2))
        self.conv13 = nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1,groups=round(r**2))
        self.conv14 = nn.Conv2d(32,round(r**4),kernel_size=3,stride=1,padding=1,groups=round(r**2))

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
    optimizer = optim.Rprop(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    net.to(device)
    hidden = net.init_hidden().to(device)
    for i in range(Epoch): 
        print("Epoch: "+ str(i))   
        batch_count = math.ceil(videoFrameCount_train/batch)
        for x in range(batch_count):
            red_train,green_train,blue_train = video_to_tensor(filepath_train,x)
            #red_train1,green_train1,blue_train1 = video_to_tensor(filepath_train,x+1)
            red_gt,green_gt,blue_gt = video_to_tensor(filepath_gt,x)
            
            for z in range(red_train.shape[0]-1):
                
                optimizer.zero_grad()
                output,hidden = net(red_train[z].to(device).unsqueeze(0),red_train[z+1].to(device).unsqueeze(0),hidden.detach())
                loss = criterion(output, red_gt[z].to(device).unsqueeze(0))
                loss.backward()
                optimizer.step()

    print("done thread Red")
    torch.save(net.state_dict(), "modelRed.pth")    

def trainingGREEN():
    net = LERN()
    optimizer = optim.Rprop(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    net.to(device)
    hidden = net.init_hidden().to(device)
    for i in range(Epoch): 
        print("Epoch: "+ str(i))   
        
        batch_count = math.ceil(videoFrameCount_train/batch)
        for x in range(batch_count):
            red_train,green_train,blue_train = video_to_tensor(filepath_train,x)
            #red_train1,green_train1,blue_train1 = video_to_tensor(filepath_train,x+1)
            red_gt,green_gt,blue_gt = video_to_tensor(filepath_gt,x)
            for z in range(green_train.shape[0]-1):
            

                optimizer.zero_grad()
                output,hidden = net(green_train[z].to(device).unsqueeze(0),green_train[z+1].to(device).unsqueeze(0),hidden.detach())
                loss = criterion(output, green_gt[z].to(device).unsqueeze(0))
                loss.backward()
                optimizer.step()

    print("done thread Green")
    torch.save(net.state_dict(), "modelGreen.pth")    

    

def trainingBLUE():
    net = LERN()
    optimizer = optim.Rprop(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    net.to(device)
    hidden = net.init_hidden().to(device)
    for i in range(Epoch): 
        print("Epoch: "+ str(i))   
        
        batch_count = math.ceil(videoFrameCount_train/batch)
        for x in range(batch_count):
            red_train,green_train,blue_train = video_to_tensor(filepath_train,x)
            #red_train1,green_train1,blue_train1 = video_to_tensor(filepath_train,x+1)
            red_gt,green_gt,blue_gt = video_to_tensor(filepath_gt,x)
            for z in range(blue_train.shape[0]-1):
            
                optimizer.zero_grad()
                output,hidden = net(blue_train[z].to(device).unsqueeze(0),blue_train[z+1].to(device).unsqueeze(0),hidden.detach())
                loss = criterion(output, blue_gt[z].to(device).unsqueeze(0))
                loss.backward()
                optimizer.step()
            print(" Percentage done: "+str(x/batch_count*100)+"            Time remaining:" + str(round((batch_count-x)*(time.time()-start)/(x+1)/60)*Epoch) + " minutes"     + "   Error: " + str(loss.item()))
    print("done thread Blue")
    torch.save(net.state_dict(), "modelBlue.pth")    

#forwards pass and save video to file
def testing():
    #device = torch.device("")
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
    
    out = cv2.VideoWriter('E:/output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), videoFPS_train, (videoWidth_train*r,videoHeight_train*r))
    batch_count = math.ceil(videoFrameCount_train/batch)
    
    for x in range(batch_count):

        red_train,green_train,blue_train = video_to_tensor(filepath_train,x)
        #red_train1,green_train1,blue_train1 = video_to_tensor(filepath_train,x+1)
        
        for z in range(red_train.shape[0]-1):
            output_red,hidden_red = net_red(red_train[z].to(device).unsqueeze(0),red_train[z+1].to(device).unsqueeze(0),hidden_red.detach())
            output_green,hidden_green = net_green(green_train[z].to(device).unsqueeze(0),green_train[z+1].to(device).unsqueeze(0),hidden_green.detach())
            output_blue,hidden_blue = net_blue(blue_train[z].to(device).unsqueeze(0),blue_train[z+1].to(device).unsqueeze(0),hidden_blue.detach())
        
            output_red =  output_red.detach().cpu().numpy()
            output_green = output_green.detach().cpu().numpy()
            output_blue = output_blue.detach().cpu().numpy()
            #save as mp4 video using cv2
            output = np.zeros((videoHeight_train*r,videoWidth_train*r,3), np.uint8)
            output[:,:,0] = output_red
            output[:,:,1] = output_green
            output[:,:,2] = output_blue
            out.write(output)
        #display video
            cv2.imshow('frame',output)
            cv2.waitKey(1)
            
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
        
        print(" Percentage done: "+str(x/batch_count*100)+"            Time remaining:" + str(round((batch_count-x)*(time.time()-start)/(x+1)/60)) + " minutes")
        

    

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

