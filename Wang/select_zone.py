import cv2

cap = cv2.VideoCapture(r"D:\Lab IC\dataThuVien_03122021\data_for_tracking\D3_7_1_2022_sang.mp4")
img = cap.read()[1]
rois = cv2.selectROIs("image", img, showCrosshair=False, fromCenter=False)
f = open("../spot_file/d3.txt", "w+")
for roi in rois:
    x, y, w, h = roi
    f.write(f"{x},{y},{w},{h}\n")
cv2.imshow("image", img)
print(rois)
