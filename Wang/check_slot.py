import cv2

spot_dict_d3 = {}
spot_dict_d5 = {}
with open("spot_file/d5.txt", "r") as f:
    for index, line in enumerate(f.readlines()):
        if not line.strip():
            continue
        x, y, w, h = list(map(int, line.split(",")))
        spot_dict_d5[index + 1] = [x, y, w, h, 0]

with open("spot_file/d3.txt", "r") as f:
    for index, line in enumerate(f.readlines()):
        if not line.strip():
            continue
        x, y, w, h = list(map(int, line.split(",")))
        spot_dict_d3[index + 1] = [x, y, w, h, 0]


def check_slot(spot_dict, id_dict, img):
    slot_return_dict = {}
    for slot, slot_box in spot_dict.items():
        x, y, w, h = slot_box
        busy = False
        for id, bbox in id_dict.items():
            x_center, y_center = bbox
            if x < x_center < x + w and y < y_center < y + h:
                slot_return_dict[slot] = [1, id]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, f"{slot} id: {id}", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 2)
                busy = True
                break
        if not busy:
            slot_return_dict[slot] = [0, None]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{slot}", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, slot_return_dict
