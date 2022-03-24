# rois = []
# with open("spot_file/nga_tu_quang_trung.txt", "r") as f:
#     lines = f.read().split("\n")
#     for line in lines:
#         if not line.strip():
#             continue
#         x, y, w, h = map(int, line.split(","))
#         rois.append([x, y, w, h])


# def get_line(x, y, w, h):
#     x1, y1 = x, y
#     x2, y2 = x + w, y + h
#     raito = x2 / x1
#     y_change = y1 * raito - y2
#     b = y_change / (raito - 1)
#     a = (y1 - b) / x1
#     return a, b


def object_counting_up_d3(old_dict, new_dict):
    for key, value in old_dict.items():
        try:
            x_old, y_old, _, _, _ = value
            x_new, y_new, _, _, _ = new_dict[key]
            if y_new <= 58 <= y_old:
                return key
        except:
            pass
    return None


def object_counting_down_d5(old_dict, new_dict):
    for key, value in old_dict.items():
        try:
            x_old, y_old, _, _, _ = value
            x_new, y_new, _, _, _ = new_dict[key]
            if x_new >= 1060 > x_old and 200 <= y_new <= 239 + 325:
                return key
        except:
            pass
    return None
