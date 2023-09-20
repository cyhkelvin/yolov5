#!/usr/bin/python3
import cv2
import numpy as np


class ProjectionControl():
    def __init__(self, pts1=None, w=405, h=525):
        self.pt1 = np.float32([])
        self.pts1 = pts1
        self.set_aspect_ratio(w, h)
        self.pixel_per_unit = 40
        self.map_img = self.draw_grid(np.zeros((h, w, 3), np.uint8))

    def set_ar_width(self, ar_width):
        self.set_aspect_ratio(ar_width, self.ar_height)

    def set_ar_height(self, ar_height):
        self.set_aspect_ratio(self.ar_width, ar_height)

    def set_aspect_ratio(self, w, h):
        self.ar_width = w
        self.ar_height = h
        self.pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    def set_pixel_per_unit(self, pixel_per_unit):
        self.pixel_per_unit = pixel_per_unit

    def set_pts1(self, lu, ru, ld, rd):
        # (x, y) of leftup, rightup, leftdown, rightdown points
        self.pts1 = np.float32([lu, ru, ld, rd])

    def set_pts1_bb(self, xywh):
        self.pts1 = np.float32([xywh[:2], \
                               [xywh[0]+xywh[2], xywh[1]], \
                               [xywh[0], xywh[1]+xywh[3]], \
                               [xywh[0]+xywh[2], xywh[1]+xywh[3]]])

    def draw_grid(self, image=None, color=(255, 255, 255)):
        """Draw grid lines in 2D view"""
        if image is None:
            image = np.zeros((self.ar_height, self.ar_width), np.uint8)
        row = int(self.ar_height / self.pixel_per_unit)
        column = int(self.ar_width / self.pixel_per_unit)
        for r in range(row + 1):
            y = r * self.pixel_per_unit
            cv2.line(image, (0, y), (self.ar_width, y), color, 2)
        for c in range(column + 1):
            x = c * self.pixel_per_unit
            cv2.line(image, (x, 0), (x, self.ar_height), color, 2)
        return image

    def draw_points(self, cx, cy, text=None, image=None):
        if image is None:
            image = self.map_img
        if self.pts1 is not None:
            cx_2d, cy_2d = ProjectionControl.get_2d_point((cx, cy), self.pts1, self.pts2)
            if text is None:
                image = self.draw_2d_points(image, cx_2d, cy_2d)
            else:
                image = self.draw_2d_points(image, cx_2d, cy_2d, text)
        return image
        
    def draw_2d_points(self, image, cx_2d, cy_2d, text=None, color=(255,255,0), p_color=(255,255,255)):
        """Draw filled circle on detected object in black 2d space"""
        # Compute bird's eye coordinates
        b_eye_x = int(cx_2d - (cx_2d % self.pixel_per_unit))
        b_eye_y = int(cy_2d - (cy_2d % self.pixel_per_unit))
        if text:
            cv2.putText(image, text, (int(b_eye_x-10), int(b_eye_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Fill-in the bird's eye box where a detected person is currently located
        pt1 = b_eye_x, b_eye_y
        pt2 = (int(pt1[0] + self.pixel_per_unit), int(pt1[1] + self.pixel_per_unit))
        cv2.rectangle(image, pt1, pt2, color, -1)

        # draw a circle indicates the location of a detected person in the bird's eye view
        cv2.circle(image, (cx_2d, cy_2d), 10, p_color, -1)
        return image

    @staticmethod
    def get_2d_point(centroid, points1, points2):
        """Get x and y coordinates that will be used in tracker module"""
        # compute point in 2D map
        # calculate matrix Homo
        homo, status = cv2.findHomography(points1, points2)
        axis = np.array([centroid], dtype='float32') # provide a point you wish to map
        axis = np.array([axis])
        points_out = cv2.perspectiveTransform(axis, homo) # finally, get the mapping
        new_x = int(points_out[0][0][0]) #points at the warped image
        new_y = int(points_out[0][0][1]) #points at the warped image
        return new_x, new_y

    @staticmethod
    def get_grid_position(position, pixel_per_unit=40):
        """Get grid coordinate"""
        grid_x = int((position[0] / pixel_per_unit) + 0.50)
        grid_y = int((position[1] / pixel_per_unit) + 0.50)
        return grid_x, grid_y
