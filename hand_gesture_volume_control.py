import cv2
import mediapipe as mp
import pyautogui 

x1 = y1 = x2 = y2 = 0
webcam = cv2.VideoCapture(0)
hand = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils #object to draw objects on hand

while True:
    _,image=webcam.read()
    image=cv2.flip(image,1)
    frame_height,frame_width,_=image.shape
    rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    output= hand.process(rgb_image)
    capt_hands=output.multi_hand_landmarks #collected all hands

    #draw points on all hands for hand in capt_hands
    if capt_hands:
        for hand_ in capt_hands:
            drawing_utils.draw_landmarks(image,hand_)
            landmarks=hand_.landmark
            
            for id, landmark in enumerate(landmarks):
                x=int(landmark.x * frame_width)
                y=int(landmark.y * frame_height)

                if id==8: #forefinger, yellow
                    cv2.circle(img=image,center=(x,y),radius=8,color=(0,255,255),thickness=3)
                    x1 = x
                    y1 = y

                if id==4: #thumb, red
                    cv2.circle(img=image,center=(x,y),radius=8,color=(0,0,255),thickness=3)
                    x2 = x
                    y2 = y
        #calculate distance between points 
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2)**(0.5)//4
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),5)
        if distance >50:
            pyautogui.press("volumeup")
        else:
            pyautogui.press("volumedown")



    cv2.imshow("Hand Gesture Volume Control",image)
    key=cv2.waitKey(10)
    if key==27:
        break
webcam.release()
cv2.destroyAllWindows()
