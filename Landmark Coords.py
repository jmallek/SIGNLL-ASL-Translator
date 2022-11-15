import cv2
import mediapipe as mp
import pandas as pd

def to_landmarks(file_name):
  # load image
  image = cv2.imread(file_name)

  # initialize hand module
  mpHands = mp.solutions.hands
  hands = mpHands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

  # convert image from BGR to RGB, flip image, and process
  results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

  # return DataFrame containing normalized x, y, z coordinates
  hand_data = []
  for hand_landmark in results.multi_hand_landmarks:
    for point in mpHands.HandLandmark:
      normalizedLandmark = hand_landmark.landmark[point]
      x, y, z = normalizedLandmark.x, normalizedLandmark.y, normalizedLandmark.z
      hand_data.append({"x":x, "y":y, "z":z})
  return pd.DataFrame(hand_data)
