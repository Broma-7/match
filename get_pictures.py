import cv2
cap = cv2.VideoCapture(1)
count = 0
while True:
    _,frame = cap.read()
    cv2.imshow('cap',frame)
    c = cv2.waitKey(1) & 0xFF
    if c == ord('q'):
        break
    elif c == ord('s'):
        cv2.imwrite(f'./Medecine/{count}.jpg',frame)
        print(f'./image/{count}.jpg is down!')
        count += 1