import cv2

cap = cv2.VideoCapture(r"D:\cam_thu_vien\17_3_2022\D3\2022-03-17\D3_cut.mp4")
img = cap.read()[1]
img = cv2.resize(img, (1280, 720))
rois = cv2.selectROIs("image", img, showCrosshair=False, fromCenter=False)
# f = open("../spot_file/d5.txt", "w+")
for roi in rois:
    x, y, w, h = roi
    # f.write(f"{x},{y},{w},{h}\n")
cv2.imshow("image", img)
print(rois)
