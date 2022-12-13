# Chương trình nhận dạng Giới tính là Male hay Female; 
# Nhận dạng tuổi với 8 phân nhóm:  (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100)
# Sử dụng CNN (Mạng nơ-ron tích chập - convolutional neural network) huấn luyện nhận dạng và phân loại hình ảnh
#Chương trình sử dụng một tập dữ liệu (dataset) gồm  26,580 bức ảnh; Mỗi bức ảnh đã được gán nhãn là male/female và 1 trong 8 nhóm tuối.

#File opencv_face_detector.pbtxt và File opencv_face_detector_uint8.pb: File định dạng text và nhị phân, là 2 file của TensorFlow
#2 file này sử dụng để chạy mô hình huấn luyện phát hiện khuôn mặt người trong ảnh, trong file có lưu định nghĩa đồ thi và các trọng số sau huấn luyện
#age_deploy.prototxt và gender_deploy.prototxt: là kiến ​​trúc mô hình cho mô hình phát hiện Tuổi và Giới tính 
#Là các tệp văn bản thuần túy có cấu trúc giống JSON chứa tất cả các định nghĩa của lớp mạng thần kinh

#age_net.caffemodel và gender_net.caffemodel: Lưu trọng số mô hình được huấn luyện để phát hiện Tuổi và Giới tính 

import cv2
import math
import argparse


#Hàm phát hiện tọa độ khuôn mặt người trong bức hình
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    
    #Xây dựng đốm màu và chuyển qua cho mạng CNN để nhận diện khuôn mặt
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    
    #Vòng lặp để trích xuất tọa độ khuôn mặt
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

# Xác định các biến trọng số và kiến ​​trúc cho các mô hình phát hiện khuôn mặt, tuổi tác và giới tính
faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

#Khởi tạo các giá trị trung bình của mô hình và các phân lớp Tuổi và Giới tính 
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

#Sử dụng các mô hình phát hiện khuôn mặt, tuổi và giới tính đã khai báo ở trên
faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

# Kích hoạt camera nếu không có hình ảnh nhận dạng
video=cv2.VideoCapture(args.image if args.image else 0)
padding=20


while cv2.waitKey(1)<0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
        
            # Tạo các đốm màu 4 chiều và chuyển các đốm màu để nhận dạng Giới tính và Tuổi
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
