import cv2

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
while True:
    success,frame = video.read()

    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grey)
    

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),color=(255,0,0),thickness=3)
    
    eyes=eye_cascade.detectMultiScale(grey)
    for (x,y,w,h) in eyes:
        cv2.circle(frame,(x,y),(x+w,y+h),color=(0,255,0),thickness=3)


    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF ==27:
        break
video.release()
cv2.destroyAllWindows()