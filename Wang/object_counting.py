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

class ObjectCounting():
    def __init__(self, new_dict, old_dict, H_limit, W_limit):
        self.new_dict = new_dict
        self.old_dict = old_dict
        self.H_limit = H_limit
        self.W_limit = W_limit

    def object_counting_up(self):
        for key, value in self.old_dict.items():
            try:
                x_old, y_old = value
                x_new, y_new = self.new_dict[key]
                if y_new <= self.H_limit < y_old and self.W_limit[0] < x_new < self.W_limit[1]:
                    return key
            except:
                pass
        return None

    def object_counting_down(self):
        for key, value in self.old_dict.items():
            try:
                x_old, y_old = value
                x_new, y_new = self.new_dict[key]
                if y_new > self.H_limit >= y_old and self.W_limit[0] < x_new < self.W_limit[1]:
                    return key
            except:
                pass
        return None
