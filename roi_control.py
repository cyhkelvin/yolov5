import cv2
from shapely.geometry import Point, Polygon


class ROIControl:
    def __init__(self, mode='rect'):
        self.status = 'init'  # init, select, track
        self.new_init = False
        self.mode = mode
        self.mode_error_msg = f'Invalid mode in ROIControl: {self.mode}'
        if mode == 'poly':
            self.pts = list()
            self.cur_pt = (-1, -1)
            self.num_pts = 4  # set -1 to accept any number of points
        elif self.mode == 'rect':
            self.target_tl = (-1, -1)
            self.target_br = (-1, -1)
        else:
            raise(self.mode_error_msg)

    def mouse_callback(self, event, x, y, flags, param):
        if self.mode == 'poly':
            if event == cv2.EVENT_LBUTTONDOWN and self.status == 'init':
                self.pts = [[x, y]]
                self.cur_pt = (x, y)
                self.status = 'select'
            elif event == cv2.EVENT_MOUSEMOVE and self.status == 'select':
                self.cur_pt = (x, y)
            elif event == cv2.EVENT_LBUTTONDOWN and self.status == 'select':
                self.pts.append(list(self.cur_pt))
                self.cur_pt = (x, y)
                if len(self.pts) == self.num_pts:
                    self.cur_pt = (-1, -1)
                    self.status = 'init'
                    self.new_init = True
            elif event == cv2.EVENT_RBUTTONDOWN and self.status == 'select':
                if len(self.pts) >= 3 and self.num_pts == -1:
                    self.cur_pt = (-1, -1)
                    self.status = 'init'
                    self.new_init = True
        elif self.mode == 'rect':
            if event == cv2.EVENT_LBUTTONDOWN and self.status == 'init':
                self.target_tl = (x, y)
                self.target_br = (x, y)
                self.status = 'select'
            elif event == cv2.EVENT_MOUSEMOVE and self.status == 'select':
                self.target_br = (x, y)
            elif event == cv2.EVENT_LBUTTONDOWN and self.status == 'select':
                self.target_br = (x, y)
                self.status = 'init'
                self.new_init = True
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.reset_bb()
                self.status = 'init'
        else:
            raise(self.mode_error_msg)

    def get_tl(self):
        return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

    def get_br(self):
        return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

    def get_bb(self):
        if self.mode == 'poly':
            ### bb = [pt1, pt2, pt3, pt4] ###
            if len(self.pts) < self.num_pts:
                bb = [*self.pts, list(self.cur_pt)]
            else:  # self.num_pts == -1
                bb = self.pts
        elif self.mode == 'rect':
            tl = self.get_tl()
            br = self.get_br()
            ### bb = [x, y, w, h] ###
            bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
        else:
            return None
        return bb

    def reset_bb(self):
        if self.mode == 'poly':
            self.pts = list()
            self.cur_pt = (-1, -1)
        elif self.mode == 'rect':
            self.target_tl = (-1, -1)
            self.target_br = (-1, -1)
        else:
            raise(self.mode_error_msg)

    def pt_in_roi(self, x, y):
        cx, cy = int(x), int(y)
        roi = self.get_bb()
        if self.num_pts > 0 and len(roi) != self.num_pts:
            return False
        if self.mode == 'rect':  # roi: [x, y, w, h]
            if cx < roi[0]:
                return False
            elif cx > roi[0] + roi[2]:
                return False
            elif cy < roi[1]:
                return False
            elif cy > roi[1] + roi[3]:
                return False
            else:
                return True
        elif self.mode == 'poly':  # roi: pts
            point = Point(cx, cy)
            poly = Polygon([tuple(i) for i in roi])
            return point.within(poly)
        else:
            raise(self.mode_error_msg)

    def set_num_pts(self, n):
        self.num_pts = n
