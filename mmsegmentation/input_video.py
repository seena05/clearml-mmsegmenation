import cv2

cap_file = cv2.VideoCapture('data/video_data/videos/scene/test1.mp4')
print(type(cap_file))
# <class 'cv2.VideoCapture'>

print(cap_file.isOpened())
# True