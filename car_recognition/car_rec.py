from car_recognition.myNet import myNet
import torch
import cv2
import torch.nn.functional as F
import os
import numpy as np

colors = ['黑色','蓝色','黄色','棕色','绿色','灰色','橙色','粉色','紫色','红色','白色']
def init_car_rec_model(model_path,device):
    check_point = torch.load(model_path)
    cfg= check_point['cfg']  
    model = myNet(num_classes=11,cfg=cfg)
    model.load_state_dict(check_point['state_dict'])
    model.to(device) 
    model.eval()
    return model

def imge_processing(img,device):
    img = cv2.resize(img,(64,64))
    img = img.transpose([2,0,1])
    img = torch.from_numpy(img).float().to(device)
    img = img-127.5
    img = img.unsqueeze(0)
    return img
    
def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)    

def get_color_and_score(model,img,device):
    img = imge_processing(img,device)
    result = model(img)
    out =F.softmax( result)
    _, predicted = torch.max(out.data, 1)
    out=out.data.cpu().numpy().tolist()[0]
    predicted = predicted.item()
    car_color= colors[predicted]
    color_conf = out[predicted]
    # print(pic_,colors[predicted[0]])
    return  car_color,color_conf
    

if __name__ == '__main__':
    # root_file =r"/mnt/Gpan/BaiduNetdiskDownload/VehicleColour/VehicleColour/class/7"
    root_file =r"imgs"
    file_list=[]
    allFilePath(root_file,file_list)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model_path = r"/mnt/Gpan/Mydata/pytorchPorject/Car_system/car_color/color_model/0.8682285244554049_epoth_117_model.pth"
    model = init_car_rec_model(model_path,device)
    for pic_ in file_list:
        img = cv2.imread(pic_)
        # img = imge_processing(img,device)
        color,conf = get_color_and_score(model,img,device)
        print(pic_,color,conf)
      
    
     
   