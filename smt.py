import cv2
import numpy as np

points_dict = {}
with open("spot_file/d3_polygon.txt", "r") as f:
    count = 0
    points = []
    for line in f.readlines():
        line = line.strip()
        line = line.split(",")
        points.append([int(line[0]), int(line[1])])
        count += 1
        if count % 4 == 0:
            points_dict[int(count / 4)] = points
            points = []

cap = cv2.VideoCapture(r"D:\cam_thu_vien\17_3_2022\D3\2022-03-17\D3_cut.mp4")
alpha = 0.3

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    img_copy = img.copy()
    output = img.copy()
    for key, value in points_dict.items():
        pts = np.array(value)
        # pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img_copy, [pts], (0, 0, 255))
        # cv2.polylines(img, [pts], True, (0, 0, 255), 2)
    img_new = cv2.addWeighted(img_copy, alpha, output, 1 - alpha, 0)
    cv2.imshow("img", img_new)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == 32:
        cv2.waitKey(0)
