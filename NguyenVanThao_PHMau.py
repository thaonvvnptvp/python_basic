# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Import thư viện Open CV, nếu chưa có thì có thể cài đặt bằng lệnh: pip install opencv-python
import cv2

#Import thư viện numpy, nếu chưa có thì có thể cài đặt bằng lệnh: pip install numpy
import numpy as np

#Import thư viện pandas, nếu chưa có thì có thể cài đặt bằng lệnh: pip install pandas
import pandas as pd

#Import thư viện có sẵn trong Python để viết Command Line Interface
import argparse

#Tạo biến để lưu trữ thông tin file ảnh được truyền vào từ CLI
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']

#Đọc file ảnh bằng OpenCV
img = cv2.imread(img_path)

#Khai báo các biến toàn cục cần thiết
clicked = False
r = g = b = xpos = ypos = 0

#Đọc file csv với Pandas và gán tên cho từng cột
index=["color","color_name","hex","R","G","B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

#Hàm tính toán và (khoảng cách tối thiểu cho các màu) và chọn màu phù hợp nhất
def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

#Hàm lấy tọa độ x,y khi kích đúp chuột và tính toán giá trị RGB tại pixel nhấp đúp chuột
def draw_function(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r,xpos,ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)
       
#Tạo một cửa sổ chứa file ảnh
cv2.namedWindow('image')
#Thiết lập hàm gọi khi có sự kiện kích chuột lên file ảnh
cv2.setMouseCallback('image',draw_function)

#Khi có sự kiện kích đúp chuột, sẽ vẽ một hình chữ nhật lên ảnh và hiển thị các thông tin cần thiết
while(1):

    cv2.imshow("image",img)
    if (clicked):
   
        #Vẽ hình chữ nhật 
        cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)

        #Tạo chuỗi thông tin
        text = getColorName(r,g,b) + ' R='+ str(r) +  ' G='+ str(g) +  ' B='+ str(b)
        
        #Hiển thị chuỗi thông tin
        cv2.putText(img, text,(50,50),2,0.8,(255,255,255),2,cv2.LINE_AA)

        #Nếu màu được chọn quá sáng, sẽ hiển thị chuỗi màu đen
        if(r+g+b>=600):
            cv2.putText(img, text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
            
        clicked=False

    #Ngắt vòng lặp nếu ấn phím Escape
    if cv2.waitKey(20) & 0xFF ==27:
        break
    
cv2.destroyAllWindows()
