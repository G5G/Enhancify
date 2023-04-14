import torch
from torch import nn
import torchvision
import numpy as np
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import math
import torch.optim as optim
import time
import threading
import cv2 as cv2
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

r = 4
q = 0.75
p = 0.75
r1 = 0.5
r2 = 0.25

video = 204
learning_rate = 0.00001
epoch = 10
trainloader = None

x_train = []
y_train = []
x_test = []
y_test = []
x_val = []
y_val = []

trainloc = ""
testloc = ""
valloc = ""

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

def fetchdata(batch,shuffle):
    
    trainsetLR = ImageFolder(root=trainloc, transform=tf)
    trainsetHR = ImageFolder(root=testloc, transform=tf)
    trainsetval = ImageFolder(root=valloc, transform=tf)
    print("Loading Training set...")
    global trainloader
    trainloader = torch.utils.data.DataLoader(list(zip(trainsetLR,trainsetHR,trainsetval)),batch_size=batch, shuffle=shuffle)
    print("Done loading data :D")
    


#torch.autograd.set_detect_anomaly(True)
    
class LERN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pixel_unshuffle = nn.PixelUnshuffle(r)
        self.pixel_shuffle = nn.PixelShuffle(r)
        self.pixel_shuffle2 = nn.PixelShuffle(r**2)
        self.conv1 = nn.ConvTranspose2d(round(r**2),round(32*q),kernel_size=3,stride=1,padding=1)
        #self.concatinate = nn.Concatinate()

        #hst-1
        self.conv2 = nn.Conv2d(round(r**4),round(32*(1-q)),kernel_size=3,stride=1,padding=1)
        #

        #sra 32
        self.leaky_relu = nn.LeakyReLU(0.25)
        #3x3dsconv&relu depthwise separable convolution 
        self.conv3 = nn.ConvTranspose2d(32,round(32*r1),kernel_size=3,stride=1,padding=1,groups=round(r**2))
        #channel shuffle
        self.channel_shuffle = nn.ChannelShuffle(r)

        self.conv4 = nn.Conv2d(round(32*r1),32,kernel_size=3,stride=1,padding=1,groups=round(r**2))

        #attention block
        #self.globalaverage = nn.GlobalAveragePool2d()
        self.conv5 = nn.ConvTranspose2d(32,round(32*r2),kernel_size=3,stride=1,padding=1)

        self.conv6 = nn.Conv2d(round(32*r2),32,kernel_size=3,stride=1,padding=1)
        self.sigmoid = nn.Sigmoid()
        #
        self.conv7 = nn.ConvTranspose2d(32,round(16*p),kernel_size=3,stride=1,padding=1)

        #lrt+1
        self.conv8 = nn.Conv2d(round(r**2),round(16*(1-p)),kernel_size=3,stride=1,padding=1)

        #sra 16
        self.conv9 = nn.ConvTranspose2d(16,round(16*r1),kernel_size=3,stride=1,padding=1)
        self.conv10 = nn.Conv2d(round(16*r1),16,kernel_size=3,stride=1,padding=1)
        self.conv11 = nn.ConvTranspose2d(16,round(16*r2),kernel_size=3,stride=1,padding=1)
        self.conv12 = nn.Conv2d(round(16*r2),16,kernel_size=3,stride=1,padding=1)
        #

        #self.conv13 = nn.ConvTranspose2d(16,8,kernel_size=3,stride=1,padding=1,groups=round(r**2))
        #self.conv14 = nn.ConvTranspose2d(8,round(r**4),kernel_size=3,stride=1,padding=1,groups=round(r**2))
        self.conv13 = nn.ConvTranspose2d(16,8,kernel_size=3,stride=1,padding=1)
        self.conv14 = nn.Conv2d(8,round(r**4),kernel_size=3,stride=1,padding=1)

        #nets
        self.conv15 = nn.ConvTranspose2d(1,round(r**2),kernel_size=3,stride=1,padding=1)

        
    def init_hidden(self,videoHeight_train,videoWidth_train):
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

start = time.time()
def trainingRED():
    net = LERN()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    net.to(device)
    hidden = net.init_hidden(32,32).to(device)
    for y in range(epoch):

        for trainLR,trainHR,trainval in trainloader:  
            optimizer.zero_grad()
            trainLR = trainLR[0][:, 0, :, :].to(device)
            trainHR = trainHR[0][:,0,:,:].to(device)
            trainval = trainval[0][:,0,:,:].to(device)
            avrgloss = 0
            for i in range(trainLR.shape[0]-1):
                output,hidden = net(trainLR[i].unsqueeze(0),trainLR[i+1].unsqueeze(0),hidden.detach())
                loss = criterion(output, trainHR[i].unsqueeze(0))
                loss.backward()
                optimizer.step()
    print("done thread Red")
    torch.save(net.state_dict(), "modelRed.pth")
    print("saved the red model")    

def trainingGREEN():
    net = LERN()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    net.to(device)
    hidden = net.init_hidden(32,32).to(device)
    x = 0 
    for y in range(epoch):
        for trainLR,trainHR in trainloader:  
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
    

def trainingBLUE():
    net = LERN()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    net.to(device)
    hidden = net.init_hidden(32,32).to(device)
    x = 0 
    for y in range(epoch):
        for trainLR,trainHR in trainloader:  
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

def testing():
    filepath_train = "C:/Users/g123lietuvis5/Desktop/LR/Vid(0).mp4"
    videoFrameCount_train,videoWidth_train,videoHeight_train,videoFPS_train = get_videodetails(filepath_train)
    net_red = LERN()
    net_red.load_state_dict(torch.load("modelRed.pth"))
    net_red.to(device)
    hidden_red = net_red.init_hidden(videoHeight_train,videoWidth_train).to(device)
    net_green = LERN()
    net_green.load_state_dict(torch.load("modelGreen.pth"))
    net_green.to(device)
    hidden_green = net_green.init_hidden(videoHeight_train,videoWidth_train).to(device)
    net_blue = LERN()
    net_blue.load_state_dict(torch.load("modelBlue.pth"))
    net_blue.to(device)
    hidden_blue = net_blue.init_hidden(videoHeight_train,videoWidth_train).to(device)
    
    out = cv2.VideoWriter('C:/Users/g123lietuvis5/Desktop/LR/output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), videoFPS_train, (videoWidth_train*r,videoHeight_train*r))
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
            cv2.imshow('frame',output)
            cv2.waitKey(1)

#GUI part of the code ----------------------------------------------------------------------------


def openFolder(inp,location):
    inp.delete("1.0", tk.END)
    
    file = filedialog.askdirectory()
    inp.insert(tk.END, file)
    if(location == "train"):
        global trainloc
        trainloc = file
    elif(location == "test"):
        global testloc
        testloc = file
    elif(location == "val"):
        global valloc
        valloc = file
        
def leaveandclose(page1,page2):
    
    page2.deiconify()
    page1.destroy()
    
def generate_plot():
    #this plots the training loss
    figure = Figure(figsize=(5, 5), dpi=100, facecolor="#FF9E3D", edgecolor="#FF9E3D")
    temp = figure.add_subplot(111)
    if(len(x_train) > 0):
        temp.plot(x_train, y_train, label="Training Loss", color="red")
        temp.xlabel("Epoch")
        temp.ylabel("Loss")
    if(len(x_test) > 0):
        temp.plot(x_test, y_test, label="Testing Loss", color="blue")
    if(len(x_val) > 0):
        temp.plot(x_val, y_val, label="Validation Loss", color="green")
    return figure

def plot_update():
    fig = generate_plot()
    for widget in right_frame.winfo_children():
        widget.destroy()
    
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
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
    
    train_frame = tk.Frame(top_frame, width=700, height=50,background="#FF9E3D")
    train_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=0)
    
    test_frame = tk.Frame(top_frame, width=700, height=50,background="#FF9E3D")
    test_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=0)
    
    val_frame = tk.Frame(top_frame, width=700, height=50,background="#FF9E3D")
    val_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=0)
    
    #backbutton_frame.pack_propagate(0)
    button_back = tk.Button(backbutton_frame, text="Back",font=("Arial", 10, "bold"),background="#B7410E",fg="white", command=lambda:leaveandclose(Train_window,Main_window))
    
    button_browseTrain = tk.Button(train_frame, text="Browse",font=("Arial", 10, "bold"),background="#B7410E",fg="white", command=lambda:openFolder(textbox_train,"train"))
    button_browseTest = tk.Button(test_frame, text="Browse",font=("Arial", 10, "bold"),background="#B7410E",fg="white", command=lambda:openFolder(textbox_test,"test"))
    button_browseVal = tk.Button(val_frame, text="Browse",font=("Arial", 10, "bold"),background="#B7410E",fg="white", command=lambda:openFolder(textbox_val,"val"))
    #make a text box
    textbox_train = tk.Text(train_frame, height=1, width=60,font=("Arial", 14, "bold"),fg="black")
    textbox_test = tk.Text(test_frame, height=1, width=60,font=("Arial", 14, "bold"),fg="black")
    textbox_val = tk.Text(val_frame, height=1, width=60,font=("Arial", 14, "bold"),fg="black")
    
    label_Epoch = tk.Label(left_frame, text="Epoch:",font=("Arial", 10, "bold"),background="#FF9E3D",fg="black")
    label_LearningRate = tk.Label(left_frame, text="LearningRate:",font=("Arial", 10, "bold"),background="#FF9E3D",fg="black")
    label_Scale = tk.Label(left_frame, text="Scale:",font=("Arial", 10, "bold"),background="#FF9E3D",fg="black")
    label_BatchSize = tk.Label(left_frame, text="BatchSize:",font=("Arial", 10, "bold"),background="#FF9E3D",fg="black")
    label_Shuffle = tk.Label(left_frame, text="Shuffle:",font=("Arial", 10, "bold"),background="#FF9E3D",fg="black")

    textbox_Epoch = tk.Text(left_frame, height=1, width=8,font=("Arial", 10, "bold"),fg="black")
    textbox_LearningRate = tk.Text(left_frame, height=1, width=8,font=("Arial", 10, "bold"),fg="black")
    textbox_Scale = tk.Text(left_frame, height=1, width=4,font=("Arial", 10, "bold"),fg="black")
    textbox_BatchSize = tk.Text(left_frame, height=1, width=6,font=("Arial", 10, "bold"),fg="black")
    checkbox_Shuffle = tk.Checkbutton(left_frame,background="#FF9E3D")
    button_Train = tk.Button(left_frame, text="Train",height=1,width=10,font=("Arial", 15, "bold"),background="#B7410E",fg="white",command=lambda:plot_update())
    
    button_back.pack(side=tk.LEFT, anchor=tk.NW)
    button_browseTrain.pack(side=tk.RIGHT, anchor=tk.NE)
    button_browseTest.pack(side=tk.RIGHT,anchor=tk.NE)
    button_browseVal.pack(side=tk.RIGHT,anchor=tk.NE)
    textbox_test.pack(side=tk.TOP, anchor=tk.N)
    textbox_train.pack(side=tk.TOP, anchor=tk.N)
    textbox_val.pack(side=tk.TOP, anchor=tk.N)
    
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
    
    Train_window.geometry("700x500")
    
    
def page_test():
    print("Preparing to test x3")
    Main_window.withdraw()
    Test_window = tk.Toplevel()
    Test_window.title("Testing the Lern network")
    Test_window.configure(background="#FF9E3D")
    button_back = tk.Button(Test_window, text="Back", command=lambda:leaveandclose(Test_window,Main_window))
    button_back.pack()
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
    button_test.pack( side=tk.LEFT)
    
    #button_train.pack()
    #button_test.pack()
    
    Main_window.geometry("600x300")
    Main_window.configure(background="#FF9E3D")
    Main_window.mainloop()
        
Main()
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
#t4 = threading.Thread(target=testing)
#t4.start()
#t4.join()
print("done")

