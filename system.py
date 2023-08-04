from status import *
import sys
import cv2
import numpy as np

# import json

Mat = np.ndarray


def handle_image_mog2(image: Mat, back_sub: cv2.BackgroundSubtractorMOG2) -> Mat:
    fg_mask = back_sub.apply(image)
    blur_again = cv2.GaussianBlur(fg_mask, (5, 5), 2, 2)
    return blur_again


class System:

    def __init__(self, stream_id=0):
        self.event = Events.NULL
        self.statu = Status.NULL
        self.cap = cv2.VideoCapture(stream_id)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame = None
        self.shape1_contour_points = []
        self.shape2_contour_points = []
        self.back_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=16)
        self.area1_list = []

    def system_read(self):
        _, img = self.cap.read()
        self.frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def system_init(self):
        _, img = self.cap.read()
        self.frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def catch_center1(self):
        blur = cv2.GaussianBlur(self.frame, (7, 7), 0)
        # mog2 = handle_image_mog2(blur, self.back_sub)
        thresh = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        centers = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = centers[0] if len(centers) == 2 else centers[1]
        for c in centers:
            if 500 > cv2.contourArea(c) > 100:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    print((center_x, center_y))
                    cv2.putText(self.frame,
                                "center",
                                (center_x - 20, center_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)
                else:
                    pass

        cv2.imshow("thresh", thresh)

    def catch_center2(self):
        blur = cv2.GaussianBlur(self.frame, (7, 7), 0)
        mog2 = handle_image_mog2(blur, self.back_sub)
        thresh = cv2.threshold(mog2, 230, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        centers = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = centers[0] if len(centers) == 2 else centers[1]
        for c in centers:
            M = cv2.moments(c)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                print((center_x, center_y))
                cv2.putText(self.frame,
                            "center",
                            (center_x - 20, center_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)
            else:
                pass

        cv2.imshow("thresh", thresh)

    def detect_shape1(self):
        self.area1_list.clear()
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(self.frame, -1, kernel)
        blur = cv2.GaussianBlur(sharpened, (7, 7), 1)

        adp = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 2)

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(adp, connectivity=8))

        mask = np.zeros_like(adp)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 50:
                mask[labels == i] = 255

        # Remove the small white dots from the image
        result = cv2.bitwise_and(adp, adp, mask=cv2.bitwise_not(mask))
        contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print("正方形：")
        for obj in contours:
            # 计算轮廓内区域的面积
            area = cv2.contourArea(obj)
            if area > 58000:
                self.area1_list.append(area)
                print(self.area1_list)
                cv2.drawContours(blur, obj, -1, (255, 0, 0), 4)
                perimeter = cv2.arcLength(obj, True)  # 计算轮廓周长
                approx = cv2.approxPolyDP(obj, 0.02 * perimeter, True)  # 获取轮廓角点坐标
                if len(approx) != 4:
                    continue
                rect = cv2.minAreaRect(approx)
                (center_x, center_y), (width, height), angle = rect
                print("Center: ({}, {}), Width: {}, Height: {}, Angle: {}".format(int(center_x),
                                                                                  int(center_y),
                                                                                  int(width),
                                                                                  int(height),
                                                                                  angle))
                cv2.circle(self.frame, (int(center_x), int(center_y)), 5, (0, 0, 0), -1)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                sorted_box = box[np.argsort(box[:, 0])]
                left_box = sorted_box[:2]
                right_box = sorted_box[2:]
                left_box = left_box[np.argsort(left_box[:, 1])]
                right_box = right_box[np.argsort(right_box[:, 1])]
                sorted_box = np.array([left_box[0], right_box[0], right_box[1], left_box[1]])
                for point in sorted_box:
                    cv2.circle(self.frame, tuple(point), 5, (255, 255, 255), -1)
                print("Sorted box points: {}".format(sorted_box))

        cv2.imshow("shape1", result)
        cv2.imshow("shape Detection", self.frame)
        cv2.waitKey(1)

    def detect_shape2(self):
        gaussBlur = cv2.GaussianBlur(self.frame, (7, 7), 1)  # 高斯模糊
        imgBlur = cv2.blur(gaussBlur, (7, 7))
        imgCanny1 = cv2.Canny(imgBlur, 60, 60)  # Canny算子边缘检测
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        imgCanny2 = cv2.filter2D(imgCanny1, -1, kernel)
        imgCanny = cv2.blur(imgCanny2, (7, 7))
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓点
        for obj in contours:
            # 计算轮廓内区域的面积
            area = cv2.contourArea(obj)
            if area < 30000:
                continue
            perimeter = cv2.arcLength(obj, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(obj, 0.02 * perimeter, True)  # 获取轮廓角点坐标
            if len(approx) != 4:
                continue
            rect = cv2.minAreaRect(approx)
            (center_x, center_y), (width, height), angle = rect
            if (260 > width > 230 and 190 > height > 160) or (260 > height > 230 and 190 > width > 160):
                cv2.drawContours(self.frame, obj, -1, (255, 0, 0), 4)  # 绘制轮廓线
                print("Center: ({}, {}), Width: {}, Height: {}, Angle: {}".format(int(center_x),
                                                                                  int(center_y),
                                                                                  int(width),
                                                                                  int(height),
                                                                                  angle))
                cv2.circle(self.frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                sorted_box = box[np.argsort(box[:, 0])]
                left_box = sorted_box[:2]
                right_box = sorted_box[2:]
                left_box = left_box[np.argsort(left_box[:, 1])]
                right_box = right_box[np.argsort(right_box[:, 1])]
                sorted_box = np.array([left_box[0], right_box[0], right_box[1], left_box[1]])
                for point in sorted_box:
                    cv2.circle(self.frame, tuple(point), 5, (0, 0, 255), -1)
                print("Sorted box points: {}".format(sorted_box))

        cv2.imshow("shape2", imgCanny)
        cv2.imshow("shape Detection", self.frame)

    def status_confirm(self, time=1):
        self.system_read()
        cv2.imshow("frame", self.frame)
        print("keyboard input")
        key = cv2.waitKey(time)

        if key == ord('t'):
            self.statu = Status.NULL
            cv2.destroyAllWindows()
        elif key == ord('r'):
            self.statu = Status.REPOSITION
        elif key == ord('a'):
            self.statu = Status.CATCH_SHAPE_1
        elif key == ord('b'):
            self.statu = Status.CATCH_SHAPE_2
        elif key == ord('c'):
            self.statu = Status.CATCH_CENTER_1
        elif key == ord('d'):
            self.statu = Status.CATCH_CENTER_2
        elif key == ord('q'):
            self.statu = Status.QUIT
        else:
            pass

    def status_handle(self):
        print("status_confirm")
        match self.statu:
            case Status.NULL:
                self.status_confirm(0)
            case Status.REPOSITION:
                pass
            case Status.CATCH_SHAPE_1:
                self.detect_shape1()
            case Status.CATCH_SHAPE_2:
                self.detect_shape2()
            case Status.CATCH_CENTER_1:
                self.catch_center1()
            case Status.CATCH_CENTER_2:
                self.catch_center2()
            case Status.SERVOS_WORK:
                pass
            case Status.QUIT:
                sys.exit(0)
