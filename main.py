import torch
from torch import nn
import torchvision
import numpy as np
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
import math
import torch.optim as optim
import time
import threading
import cv2 as cv2
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


r = 4
q = 0.75
p = 0.75
r1 = 0.5
r2 = 0.25

video = 204
learning_rate = 0.00005
epoch = 10
trainloaderHR = None
trainloaderLR = None
valloaderHR = None
valloaderLR = None

x_train = []
y_train = []
x_val = []
y_val = []


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

def video_to_tensor(video_path,batch_number):
    capture = cv2.VideoCapture(video_path)
    framecount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #batch_count = math.ceil(framecount/batch)
    frame_r_list = []
    frame_g_list = []
    frame_b_list = []
    for i in range(20):
        frame_index = 20 * batch_number + i
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

tf = transforms = transforms.Compose([
    transforms.ToTensor(),  
    #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

def fetchdata(batch,shuffle,learningRate,eepoch,trainlocHR,trainlocLR,vallocHR,vallocLR):
    
    batch = batch.get("1.0", "end-1c")
    batch = int(batch)
    shuffle = shuffle.get()
    learningRate = learningRate.get("1.0", "end-1c")
    learningRate = float(learningRate)
    eepoch = eepoch.get("1.0", "end-1c")
    eepoch = int(eepoch)
    trainlocLR = trainlocLR.get("1.0","end-1c")
    trainlocHR = trainlocHR.get("1.0", "end-1c")
    vallocHR = vallocHR.get("1.0", "end-1c")
    vallocLR = vallocLR.get("1.0", "end-1c")
    #remove this----------------------------------------------------
    #batch = 10
    #shuffle = True
    #learningRate = 0.001
    #eepoch = 200
    #trainlocHR = "C:/Users/123li/Downloads/train_sharp/train/train_sharp"
    #trainlocLR = "C:/Users/123li/Downloads/train_sharp_bicubic/train/train_sharp_bicubic/X4"
    #vallocHR = "C:/Users/123li/Downloads/val_sharp/val/val_sharp"
    #vallocLR = "C:/Users/123li/Downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4"
    #-remove this---------------------------------------------------

    trainsetLR = ImageFolder(root=trainlocLR, transform=tf)
    trainsetHR = ImageFolder(root=trainlocHR, transform=tf)
    trainsetvalHR = ImageFolder(root=vallocHR, transform=tf)
    trainsetvalLR = ImageFolder(root = vallocLR, transform = tf)




    print("Loading Training set...")
    global trainloaderHR,trainloaderLR,valloaderHR,valloaderLR,learning_rate,epoch
    learning_rate = learningRate
    epoch = eepoch
    
    
    trainloaderHR = torch.utils.data.DataLoader(trainsetHR, batch_size=batch, shuffle=shuffle, num_workers=0)
    trainloaderLR = torch.utils.data.DataLoader(trainsetLR, batch_size=batch, shuffle=shuffle, num_workers=0)
    valloaderHR = torch.utils.data.DataLoader(trainsetvalHR, batch_size=batch, shuffle=shuffle, num_workers=0)
    valloaderLR = torch.utils.data.DataLoader(trainsetvalLR, batch_size=batch, shuffle=shuffle, num_workers=0)
    print("Done loading data :D")
    #print trainloader size
    #trainingRED()

    t1 = threading.Thread(target=trainingRED)
    t2 = threading.Thread(target=trainingGREEN)
    t3 = threading.Thread(target=trainingBLUE)

    t1.start()
    t2.start()
    t3.start()

    #t1.join()
    #t2.join()
    #t3.join()
    
    


torch.autograd.set_detect_anomaly(True)
    
class LERN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pixel_unshuffle = nn.PixelUnshuffle(r)
        self.pixel_shuffle = nn.PixelShuffle(r)
        self.pixel_shuffle2 = nn.PixelShuffle(r**2)
        self.conv1 = nn.ConvTranspose2d(round(r**2),round(32*q),kernel_size=3,stride=1,padding=1)

        #hst-1
        self.conv2 = nn.Conv2d(round(r**4),round(32*(1-q)),kernel_size=3,stride=1,padding=1)
        #

        #sra 32
        self.leaky_relu = nn.LeakyReLU(0.25)

        #depthwise separable convolution

        self.conv3_dep = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,groups=32)#depthwise convolution
        self.conv3_point = nn.Conv2d(32,round(32*r1),kernel_size=1,stride=1,padding=0)#pointwise convolution
        #channel shuffle
        #self.channel_shuffle = nn.ChannelShuffle(3)

        self.conv4_dep = nn.Conv2d(round(32*r1),round(32*r1),kernel_size=3,stride=1,padding=1,groups=round(32*r1))#depthwise convolution
        self.conv4_point = nn.Conv2d(round(32*r1),32,kernel_size=1,stride=1,padding=0)#pointwise convolution
        #attention block
        self.globalaverage = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.ConvTranspose2d(32,round(32*r2),kernel_size=3,stride=1,padding=1)

        self.conv6 = nn.Conv2d(round(32*r2),32,kernel_size=3,stride=1,padding=1)
        self.sigmoid = nn.Sigmoid()

        self.conv7_dep = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,groups=32)#depthwise convolution
        self.conv7_point = nn.Conv2d(32,round(16*p),kernel_size=1,stride=1,padding=0)#pointwise convolution

        #lrt+1
        self.conv8 = nn.Conv2d(round(r**2),round(16*(1-p)),kernel_size=3,stride=1,padding=1)

        #sra 16
        self.conv9_dep = nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1,groups=16)#depthwise convolution
        self.conv9_point = nn.Conv2d(16,round(16*r1),kernel_size=1,stride=1,padding=0)#pointwise convolution

        self.conv10_dep = nn.Conv2d(round(16*r1),round(16*r1),kernel_size=3,stride=1,padding=1,groups=round(16*r1))#depthwise convolution
        self.conv10_point = nn.Conv2d(round(16*r1),16,kernel_size=1,stride=1,padding=0)#pointwise convolution
        self.conv11 = nn.Conv2d(16,round(16*r2),kernel_size=3,stride=1,padding=1)
        self.conv12 = nn.Conv2d(round(16*r2),16,kernel_size=3,stride=1,padding=1)
        #
        self.conv13_dep = nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1,groups=16)#depthwise convolution
        self.conv13_point = nn.Conv2d(16,8,kernel_size=1,stride=1,padding=0)#pointwise convolution

        self.linear1 = nn.Linear(8,8)
        self.conv14_dep = nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1,groups=8)#depthwise convolution
        self.conv14_point = nn.Conv2d(8,round(r**4),kernel_size=1,stride=1,padding=0)#pointwise convolution
        #nets
        
        self.conv15 = nn.Conv2d(1,round(r**2),kernel_size=3,stride=1,padding=1)

        
    def init_hidden(self,videoHeight_train,videoWidth_train):
        return torch.rand(r**4,round(videoHeight_train/r),round(videoWidth_train/r)).to(device)    
    def forward(self, x,x1,htt):        
        #hstt = self.hst
        lrt = x
        lrt1 = x1
        hstt = htt
        #top of neth
        lrt = self.pixel_unshuffle(lrt) # 1 channel
        lrt = self.conv1(lrt)
        
        hstt = self.conv2(hstt)
        lrt = torch.cat((lrt,hstt),0)
        #sra Channel 32--- 
        lrrt = lrt
        lrt = self.leaky_relu(lrt)
        lrt = self.conv3_dep(lrt)
        lrt = self.conv3_point(lrt)
        lrt = self.leaky_relu(lrt)
        
        #lrt = self.channel_shuffle(lrt)

        lrt = self.conv4_dep(lrt)
        lrt = self.conv4_point(lrt)
        #lrt = self.channel_shuffle(lrt)
        #attention block
        lrrrt = lrt
        lrt = self.globalaverage(lrt)
        lrt = self.conv5(lrt)
        lrt = self.leaky_relu(lrt)
        lrt = self.conv6(lrt)
        lrt = self.sigmoid(lrt)
        lrt = lrrrt*lrt
        lrt = lrrt+lrt
        #
        lrt = self.leaky_relu(lrt)
        lrt = self.conv7_dep(lrt)
        lrt = self.conv7_point(lrt)
        
        #lrt+1
        lrt1 = self.pixel_unshuffle(lrt1)
        lrt1 = self.conv8(lrt1)

        lrt = torch.cat((lrt,lrt1),0)

        #sra part 16
        lrrt = lrt
        lrt = self.leaky_relu(lrt)
        #lrt = self.leaky_relu(self.conv9(lrt))
        lrt = self.conv9_dep(lrt)
        lrt = self.conv9_point(lrt)
        lrt = self.leaky_relu(lrt)
        #lrt = self.channel_shuffle(lrt)
        lrt = self.conv10_dep(lrt)
        lrt = self.conv10_point(lrt)
        #lrt = self.channel_shuffle(lrt)
        lrrrt = lrt
        lrt = self.globalaverage(lrt)
        lrt = self.conv11(lrt)
        lrt = self.leaky_relu(lrt)
        lrt = self.conv12(lrt)
        lrt = self.sigmoid(lrt)
        lrt = lrrrt*lrt
        lrt = lrrt+lrt
        #
        lrt = self.leaky_relu(lrt)
        lrt = self.conv13_dep(lrt)
        lrt = self.conv13_point(lrt)

        lrt = self.leaky_relu(lrt)
        lrt = self.conv14_dep(lrt)
        lrt = self.conv14_point(lrt)
        lrt = self.leaky_relu(lrt)
        hstt = lrt

        #nets
        x = self.conv15(x)
        x = self.pixel_shuffle(x)
        lrt = self.pixel_shuffle2(lrt)
        x = lrt+x

        #self.hst = hstt

        return x,hstt

start = time.time()
def trainingRED():
    net = LERN()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.00001,amsgrad=True)
    criterion = nn.MSELoss()
    net.to(device)
    hidden = net.init_hidden(trainloaderLR.dataset[0][0][0].shape[0],trainloaderLR.dataset[0][0][0].shape[1]).to(device)
    totalit = len(trainloaderLR) * trainloaderLR.batch_size

    for y in range(epoch):
        print("Epoch: ",y,"/",epoch)
        PSNR_train = 0
        Val_loss = 0
        train_Count = 0
        
        val_Count = 0
        for (trainLR,trainHR) in zip(trainloaderLR,trainloaderHR):
            
            #calculates how many iterations are left
            print("Iterations left: ",totalit - train_Count)
            optimizer.zero_grad()
            trainLR = trainLR[0][:, 0, :, :].to(device)
            trainHR = trainHR[0][:,0,:,:].to(device)
            for i in range(trainLR.shape[0]-1):
                output,hidden = net(trainLR[i].unsqueeze(0),trainLR[i+1].unsqueeze(0),hidden.detach())
                loss = criterion(output, trainHR[i].unsqueeze(0))
                loss.backward()
                optimizer.step()
                mse = torch.mean((output - trainHR[i])**2)
                psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))
                print("PSNR train: ",psnr)
                PSNR_train += psnr
                train_Count += 1
        print("done thread Red")
        torch.save(net.state_dict(), "modelRed.pth")
        print("saved the red model")  

                #training loss calculated
        PSNR_train = PSNR_train/train_Count
        print("PSNR_Train: ",PSNR_train)
        y_train.append(PSNR_train)
        x_train.append(y)
        plot_update()
        with torch.no_grad():
            for(valLR,valHR) in zip(valloaderLR,valloaderHR):
                valLR = valLR[0][:, 0, :, :].to(device)
                valHR = valHR[0][:,0,:,:].to(device)
                for i in range(valLR.shape[0]-1):
                    output,hidden = net(valLR[i].unsqueeze(0),valLR[i+1].unsqueeze(0),hidden.detach())
                    loss = criterion(output, valHR[i].unsqueeze(0))

                    mse = torch.mean((output - valHR[i])**2)
                    psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))
                    print("PSNR Val: ",psnr)
                    Val_loss += psnr
                    val_Count += 1

                    #validation loss calculated
            Val_loss = Val_loss/val_Count
            y_val.append(Val_loss)
            x_val.append(y)
            plot_update()

        
    print("done thread Red")
    torch.save(net.state_dict(), "modelRed.pth")
    print("saved the red model")    

def trainingGREEN():
    net = LERN()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.00001,amsgrad=True)
    criterion = nn.MSELoss()
    net.to(device)
    hidden = net.init_hidden(trainloaderLR.dataset[0][0][0].shape[0],trainloaderLR.dataset[0][0][0].shape[1]).to(device)

    for y in range(epoch):
        for (trainLR,trainHR) in zip(trainloaderLR,trainloaderHR):
            optimizer.zero_grad()
            trainLR = trainLR[0][:, 1, :, :].to(device)
            trainHR = trainHR[0][:,1,:,:].to(device)
            for i in range(trainLR.shape[0]-1):
                output,hidden = net(trainLR[i].unsqueeze(0),trainLR[i+1].unsqueeze(0),hidden.detach())
                loss = criterion(output, trainHR[i].unsqueeze(0))
                loss.backward()
                optimizer.step()
        print("done thread green")
        torch.save(net.state_dict(), "modelGreen.pth") 
        print("saved the green model")   
    print("done thread green")
    torch.save(net.state_dict(), "modelGreen.pth") 
    print("saved the green model")   
    

def trainingBLUE():
    net = LERN()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.00001,amsgrad=True)
    criterion = nn.MSELoss()
    net.to(device)
    hidden = net.init_hidden(trainloaderLR.dataset[0][0][0].shape[0],trainloaderLR.dataset[0][0][0].shape[1]).to(device)

    for y in range(epoch):
        for (trainLR,trainHR) in zip(trainloaderLR,trainloaderHR): 
            optimizer.zero_grad()
            trainLR = trainLR[0][:, 2, :, :].to(device)
            trainHR = trainHR[0][:,2,:,:].to(device)
            for i in range(trainLR.shape[0]-1):
                output,hidden = net(trainLR[i].unsqueeze(0),trainLR[i+1].unsqueeze(0),hidden.detach())
                loss = criterion(output, trainHR[i].unsqueeze(0))
                loss.backward()
                optimizer.step()
        print("done thread Blue")
        torch.save(net.state_dict(), "modelBlue.pth")
        print("saved the blue model")   

    print("done thread Blue")
    torch.save(net.state_dict(), "modelBlue.pth")
    print("saved the blue model")   

def testing(Lr,HrLocation,modelLocation):
    #device = torch.device("cpu")
    
    filepath_train = Lr.get("1.0", "end-1c")
    videoFrameCount_train,videoWidth_train,videoHeight_train,videoFPS_train = get_videodetails(filepath_train)
    
    
    net_red = LERN()
    net_green = LERN()
    net_blue = LERN()
    
    net_red.load_state_dict(torch.load(modelLocation.get("1.0","end-1c")+"/modelRed.pth", map_location=torch.device('cpu')))
    net_green.load_state_dict(torch.load(modelLocation.get("1.0","end-1c")+"/modelGreen.pth", map_location=torch.device('cpu')))
    net_blue.load_state_dict(torch.load(modelLocation.get("1.0","end-1c")+"/modelBlue.pth", map_location=torch.device('cpu')))
    net_red.to(device)
    net_green.to(device)
    net_blue.to(device)

    hidden_red = net_red.init_hidden(videoHeight_train,videoWidth_train).to(device)
    hidden_green = net_green.init_hidden(videoHeight_train,videoWidth_train).to(device)
    hidden_blue = net_blue.init_hidden(videoHeight_train,videoWidth_train).to(device)
    
    out = cv2.VideoWriter((HrLocation.get("1.0","end-1c") + "/output.mp4"),cv2.VideoWriter_fourcc(*'mp4v'), videoFPS_train, (videoWidth_train*r,videoHeight_train*r))
    batch_count = math.ceil(videoFrameCount_train/20)
    
    for x in range(batch_count):
        red_train,green_train,blue_train = video_to_tensor(filepath_train,x)
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
            original = np.zeros((videoHeight_train, videoWidth_train, 3), np.uint8)
            original[:,:,0] = red_train[z].detach().cpu().numpy()
            original[:,:,1] = green_train[z].detach().cpu().numpy()
            original[:,:,2] = blue_train[z].detach().cpu().numpy()
            
            rs = cv2.resize(original, (600,400))
            rs2 = cv2.resize(output, (600,400))
            output = np.concatenate((rs,rs2), axis=1)
            
            
            cv2.putText(output, 'Original', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(output, 'upscaled', (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            resized = output / np.max(output)
            cv2.imshow('frame',resized)
            
            cv2.waitKey(1)
    out.release()
    cv2.destroyAllWindows()
#GUI part of the code ----------------------------------------------------------------------------


def openFolder(inp):
    inp.delete("1.0", tk.END)
    
    file = filedialog.askdirectory()
    inp.insert(tk.END, file)
        
def openFile(inp):
    inp.delete("1.0", tk.END)
    #open videos only
    file = filedialog.askopenfilename(filetypes = (("Video files","*.mp4"),("all files","*.*")))
    inp.insert(tk.END, file)

def leaveandclose(page1,page2):
    
    page2.deiconify() 
    page1.destroy()
    
def generate_plot():
    #this plots the training loss
    figure = Figure(figsize=(5, 5), dpi=100, facecolor="#FF9E3D", edgecolor="#FF9E3D")
    temp = figure.add_subplot(111)
    if(len(x_train) > 0):
        temp.plot(x_train, y_train, label="Training Loss", color="red")
        
    if(len(x_val) > 0):
        temp.plot(x_val, y_val, label="Validation Loss", color="green")
        
    temp.legend()
    return figure

def plot_update():
    fig = generate_plot()
    for widget in right_frame.winfo_children():
        widget.destroy()
    
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    bar = NavigationToolbar2Tk(canvas, right_frame)
    bar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)



def page_train():
    print("Preparing to train the Lern network")
    Main_window.withdraw()
    Train_window = tk.Toplevel()
    Train_window.title("Training the Lern network")
    Train_window.configure(background="#FF9E3D")
    top_frame = tk.Frame(Train_window, width=700, height=100,background="#FF9E3D")
    top_frame.pack(side=tk.TOP, fill=tk.BOTH)
    
    left_frame = tk.Frame(Train_window, width=350, height=400,background="#FF9E3D")
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=0)
    global right_frame
    right_frame = tk.Frame(Train_window, width=350, height=400,background="#FF9E3D")
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=0)
    
    backbutton_frame = tk.Frame(top_frame, width=200, height=50,background="#FF9E3D")
    backbutton_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=0)
    
    trainHR_frame = tk.Frame(top_frame, width=700, height=50,background="#FF9E3D")
    trainHR_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=0)
    
    trainLR_frame = tk.Frame(top_frame, width=700, height=50,background="#FF9E3D")
    trainLR_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=0)
    
    val_frameHR = tk.Frame(top_frame, width=700, height=50,background="#FF9E3D")
    val_frameHR.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=0)

    val_frameLR = tk.Frame(top_frame,width=700,height = 50,background="#FF9E3D")
    val_frameLR.pack(side=tk.TOP,fill=tk.BOTH,padx=10,pady=0)

    button_back = tk.Button(backbutton_frame, text="Back",font=("Arial", 10, "bold"),background="#B7410E",fg="white", command=lambda:leaveandclose(Train_window,Main_window))
    
    button_browseTrainHR = tk.Button(trainHR_frame, text="Browse HR Training Data",font=("Arial", 8, "bold"),background="#B7410E",fg="white", command=lambda:openFolder(textbox_trainHR))
    button_browseTrainLR = tk.Button(trainLR_frame, text="Browse LR Training Data",font=("Arial", 8, "bold"),background="#B7410E",fg="white", command=lambda:openFolder(textbox_trainLR))
    button_browseValHR = tk.Button(val_frameHR, text="Browse HR Validate Data",font=("Arial", 8, "bold"),background="#B7410E",fg="white", command=lambda:openFolder(textbox_valHR))
    button_browseValLR = tk.Button(val_frameLR,text="Browse LR Validate Data",font=("Arial",8,"bold"),background="#B7410E",fg="white",command=lambda:openFolder(textbox_valLR))
    #make a text box
    textbox_trainHR = tk.Text(trainHR_frame, height=1, width=60,font=("Arial", 14, "bold"),fg="black")
    textbox_trainLR = tk.Text(trainLR_frame, height=1, width=60,font=("Arial", 14, "bold"),fg="black")
    textbox_valHR = tk.Text(val_frameHR, height=1, width=60,font=("Arial", 14, "bold"),fg="black")
    textbox_valLR = tk.Text(val_frameLR, height=1, width=60,font=("Arial", 14, "bold"),fg="black")

    label_Epoch = tk.Label(left_frame, text="Epoch:",font=("Arial", 10, "bold"),background="#FF9E3D",fg="black")
    label_LearningRate = tk.Label(left_frame, text="LearningRate:",font=("Arial", 10, "bold"),background="#FF9E3D",fg="black")
    label_Scale = tk.Label(left_frame, text="Scale:",font=("Arial", 10, "bold"),background="#FF9E3D",fg="black")
    label_BatchSize = tk.Label(left_frame, text="BatchSize:",font=("Arial", 10, "bold"),background="#FF9E3D",fg="black")
    label_Shuffle = tk.Label(left_frame, text="Shuffle:",font=("Arial", 10, "bold"),background="#FF9E3D",fg="black")

    textbox_Epoch = tk.Text(left_frame, height=1, width=8,font=("Arial", 10, "bold"),fg="black")
    textbox_LearningRate = tk.Text(left_frame, height=1, width=8,font=("Arial", 10, "bold"),fg="black")
    textbox_Scale = tk.Text(left_frame, height=1, width=4,font=("Arial", 10, "bold"),fg="black")
    textbox_BatchSize = tk.Text(left_frame, height=1, width=6,font=("Arial", 10, "bold"),fg="black")
    checkStatus = tk.IntVar()
    checkbox_Shuffle = tk.Checkbutton(left_frame,background="#FF9E3D",variable=checkStatus)
    button_Train = tk.Button(left_frame, text="Train",height=1,width=10,font=("Arial", 15, "bold"),background="#B7410E",fg="white",command=lambda:fetchdata(textbox_BatchSize,checkStatus,textbox_LearningRate,textbox_Epoch,textbox_trainHR,textbox_trainLR,textbox_valHR,textbox_valLR))
    
    button_back.pack(side=tk.LEFT, anchor=tk.NW)
    button_browseTrainHR.pack(side=tk.RIGHT, anchor=tk.NE)
    button_browseTrainLR.pack(side=tk.RIGHT,anchor=tk.NE)
    button_browseValHR.pack(side=tk.RIGHT,anchor=tk.NE)
    button_browseValLR.pack(side=tk.RIGHT,anchor=tk.NE)
    textbox_trainHR.pack(side=tk.TOP, anchor=tk.N)
    textbox_trainLR.pack(side=tk.TOP, anchor=tk.N)
    textbox_valHR.pack(side=tk.TOP, anchor=tk.N)
    textbox_valLR.pack(side=tk.TOP,anchor=tk.N)
    
    label_Epoch.grid(row=0, column=0, sticky="W", padx=0, pady=20)
    textbox_Epoch.grid(row=0, column=1, sticky="E", padx=0, pady=20)
    label_LearningRate.grid(row=1, column=0, sticky="W", padx=0, pady=20)
    textbox_LearningRate.grid(row=1, column=1, sticky="E", padx=0, pady=20)
    label_Scale.grid(row=2, column=0, sticky="W", padx=0, pady=20)
    textbox_Scale.grid(row=2, column=1, sticky="E", padx=0, pady=20)
    label_BatchSize.grid(row=3, column=0, sticky="W", padx=0, pady=20)
    textbox_BatchSize.grid(row=3, column=1, sticky="E", padx=0, pady=20)
    label_Shuffle.grid(row=4, column=0, sticky="W", padx=0, pady=20)
    checkbox_Shuffle.grid(row=4, column=1, sticky="E", padx=0, pady=20)
    button_Train.grid(row=5, column = 3,sticky="W", padx=60, pady=40)
    
    plot_update()
    
    Train_window.geometry("700x700")
    
    
def page_test():
    print("Preparing to test x3")
    Main_window.withdraw()
    Test_window = tk.Toplevel()
    Test_window.title("Testing the Lern network")
    Test_window.configure(background="#FF9E3D")
    top_frame = tk.Frame(Test_window, width=700, height=100,background="#FF9E3D")
    top_frame.pack(side=tk.TOP, fill=tk.BOTH)
    backbutton_frame = tk.Frame(top_frame, width=200, height=50,background="#FF9E3D")
    backbutton_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=0)

    LRvid_frame = tk.Frame(top_frame, width=700, height=50,background="#FF9E3D")
    LRvid_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=0)
    HRvid_frame = tk.Frame(top_frame, width=700, height=50,background="#FF9E3D")
    HRvid_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=0)
    model_frame = tk.Frame(top_frame,width= 700, height=50,background="#FF9E3D")
    model_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=0)
    run_frame = tk.Frame(top_frame, width=700, height=50,background="#FF9E3D")
    run_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=0)

    textbox_loadLR = tk.Text(LRvid_frame, height=1, width=60,font=("Arial", 10, "bold"),fg="black")
    textbox_outputHR = tk.Text(HRvid_frame, height=1, width=60,font=("Arial", 10, "bold"),fg="black")
    textbox_model = tk.Text(model_frame, height=1, width=60,font=("Arial", 10, "bold"),fg="black")
    
    button_browseLRvid = tk.Button(LRvid_frame, text="Browse Video for upscaling",font=("Arial", 8, "bold"),background="#B7410E",fg="white", command=lambda:openFile(textbox_loadLR))
    button_browseHRout = tk.Button(HRvid_frame, text="Browse output folder",font=("Arial", 8, "bold"),background="#B7410E",fg="white", command=lambda:openFolder(textbox_outputHR))
    button_browseModel = tk.Button(model_frame, text="Browse Model folder",font=("Arial", 8, "bold"),background="#B7410E",fg="white", command=lambda:openFolder(textbox_model))
    button_back = tk.Button(backbutton_frame, text="Back",font=("Arial", 10, "bold"),background="#B7410E",fg="white", command=lambda:leaveandclose(Test_window,Main_window))
    button_run = tk.Button(run_frame, text="Run",font=("Arial", 10, "bold"),background="#B7410E",fg="white", command=lambda:testing(textbox_loadLR,textbox_outputHR,textbox_model))
    button_back.pack(side=tk.LEFT, anchor=tk.NW)
    button_browseLRvid.pack(side=tk.RIGHT, anchor=tk.NW)
    textbox_outputHR.pack(side=tk.LEFT, anchor=tk.NW)
    button_browseHRout.pack(side=tk.RIGHT, anchor=tk.NW)
    textbox_loadLR.pack(side=tk.LEFT, anchor=tk.NW)
    button_browseModel.pack(side=tk.RIGHT, anchor=tk.NW)
    textbox_model.pack(side=tk.LEFT, anchor=tk.NW)
    button_run.pack(side=tk.RIGHT, anchor=tk.NW)

    
    
    Test_window.geometry("700x500")
    


def Main():
    global Main_window
    
    Main_window = tk.Tk()
    
    Main_window.title("video Super Resolution")
    label_title = tk.Label(Main_window, text="Video Super Resolution",background="#FF9E3D", fg="white",font=("Arial", 20, "bold"),height=2,width=20)  
    button_train = tk.Button(Main_window, text="Train the Lern network", font=("Arial", 15, "bold"),background="#B7410E",fg="white",command=page_train)
    button_test = tk.Button(Main_window, text="Test the Lern network", font=("Arial", 15, "bold"),background="#B7410E",fg="white", command=page_test)
    label_title.pack()
    button_train.pack(side=tk.RIGHT)
    button_test.pack(side=tk.LEFT)
    
    Main_window.geometry("600x300")
    Main_window.configure(background="#FF9E3D")
    Main_window.mainloop()
        
Main()

print("done")

