import cv2
import numpy as np

points_dict_d3 = {}
with open("spot_file/d3_polygon.txt", "r") as f:
    count = 0
    points = []
    for line in f.readlines():
        line = line.strip()
        line = line.split(",")
        points.append([int(line[0]), int(line[1])])
        count += 1
        if count % 4 == 0:
            points_dict_d3[int(count / 4)] = points
            points = []

points_dict_d5 = {}
with open("spot_file/d5_polygon.txt", "r") as f:
    count = 0
    points = []
    for line in f.readlines():
        line = line.strip()
        line = line.split(",")
        points.append([int(line[0]), int(line[1])])
        count += 1
        if count % 4 == 0:
            points_dict_d5[int(count / 4)] = points
            points = []


def draw_polygon_d3(img, key, color):
    points = points_dict_d3[key]
    pts = np.array(points)
    cv2.fillPoly(img, [pts], color)


def draw_polygon_d5(img, key, color):
    points = points_dict_d5[key]
    pts = np.array(points)
    cv2.fillPoly(img, [pts], color)
