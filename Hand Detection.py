import cv2
import mediapipe as mp
import time

def get_sign(hand_data):
    if hand_data: #if the hand is in frame
        if nonthumbs_aligned(hand_data) and hand_data[20][1] > hand_data[19][1] and hand_data[16][1] > hand_data[15][1] and hand_data[12][1] > hand_data[11][1] and hand_data[8][1] > hand_data[7][1]:
            if hand_data[4][1] < hand_data[8][1] and hand_data[4][1] > hand_data[6][1] and hand_data[4][0]<hand_data[8][0]:
                print('s '+str(time.time()))
            elif hand_data[4][0] > hand_data[6][0]: print('a '+str(time.time()))
            elif hand_data[4][1] > hand_data[12][1] and hand_data[4][1] > hand_data[16][1] and hand_data[4][0] > hand_data[20][0] and hand_data[4][0] < hand_data[8][0]: print('e '+str(time.time()))
        # line 12: all fingers pointing straight up
        elif nonthumbs_aligned(hand_data) and hand_data[20][1] < hand_data[19][1] and hand_data[16][1] < hand_data[15][1] and hand_data[12][1] < hand_data[11][1] and hand_data[8][1] < hand_data[7][1]:
            if hand_data[4][1] > hand_data[20][1] and hand_data[4][0] < hand_data[8][0] and hand_data[4][0] > hand_data[20][0]:
                print('b '+str(time.time()))
        # print(str(hand_data[8][0])+" "+str(hand_data[8][1]))

def nonthumbs_aligned(hand_data):
    if hand_data:
        if hand_data[18][0] < hand_data[14][0] and hand_data[14][0] < hand_data[10][0] and hand_data[10][0]<hand_data[6][0]:
            return True
    return False

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
first = True
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        hand_data = []
        for handLms in results.multi_hand_landmarks:
            for point in mpHands.HandLandmark:
                normalizedLandmark = handLms.landmark[point]
                x, y, z = normalizedLandmark.x, normalizedLandmark.y, normalizedLandmark.z
                hand_data.append((x, y, z))
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                #if id ==0:
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
                cv2.putText(img, str(id), (cx-20,cy), cv2.FONT_HERSHEY_PLAIN, 1, (26, 31, 120), 1)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)               
        get_sign(hand_data)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)