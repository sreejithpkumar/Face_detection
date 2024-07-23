import cv2
import time
i = 1
model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vebcam = cv2.VideoCapture(0)
while True:
    successfull_frame_read, img = vebcam.read()
    if successfull_frame_read == True:

        black_and_white_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face_co = model.detectMultiScale(black_and_white_img, minNeighbors=12)
        s = len(face_co)
        for n in range(s):
            x = face_co[n][0]
            y = face_co[n][1]
            w = face_co[n][2]
            h = face_co[n][3]
            cv2.rectangle(img,(x, y),(x+w,y+h),(0,255,0),2)

        cv2.imshow('Face Recognition',img)

        key = cv2.waitKey(1)
        if key==81 or key==113:
            break

       # if s != 0:
            cv2.imwrite(f"D:\Image\image{i}.jpg", img)
            i = i+1
            if i%10 == 0:
                break


print("Program completed")
vebcam.release()
cv2.destroyAllWindows()