import cv2

# Reading video
video = cv2.VideoCapture(0)  # 0 displays the webcam

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

while True:
    success, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 0), thickness=3)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            center_x = x + ex + ew // 2
            center_y = y + ey + eh // 2
            radius = int((ew + eh) / 4)
            cv2.circle(frame, (center_x, center_y), radius, color=(255, 255, 255), thickness=3)

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Escape key
        break

video.release()
cv2.destroyAllWindows()
