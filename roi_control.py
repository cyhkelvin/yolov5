import cv2


class ROIControl:
    def __init__(self):
        self.mode = 'init'  # init, select, track
        self.target_tl = (-1, -1)
        self.target_br = (-1, -1)
        self.new_init = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.mode == 'init':
            self.target_tl = (x, y)
            self.target_br = (x, y)
            self.mode = 'select'
        elif event == cv2.EVENT_MOUSEMOVE and self.mode == 'select':
            self.target_br = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN and self.mode == 'select':
            self.target_br = (x, y)
            self.mode = 'init'
            self.new_init = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.reset_bb()
            self.mode = 'init'

    def get_tl(self):
        return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

    def get_br(self):
        return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

    def get_bb(self):
        tl = self.get_tl()
        br = self.get_br()
        bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
        return bb

    def reset_bb(self):
        self.target_tl = (-1, -1)
        self.target_br = (-1, -1)

    def pt_in_bb(self, x, y):
        bb = self.get_bb()
        if x < bb[0]:
            return False
        elif x > bb[0] + bb[2]:
            return False
        elif y < bb[1]:
            return False
        elif y > bb[1] + bb[3]:
            return False
        else:
            return True
