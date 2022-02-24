import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For static images:
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def get_face_key_point(image):
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        h, w, _ = image.shape
        print(image.shape)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.

        annotated_image = image.copy()
        for detection in results.detections:
            print("Nose tip:")

            xmin, ymin, width, height = (
                int(detection.location_data.relative_bounding_box.xmin * w),
                int(detection.location_data.relative_bounding_box.ymin * h),
                int(detection.location_data.relative_bounding_box.width * w),
                int(detection.location_data.relative_bounding_box.height * h),
            )
            cv2.rectangle(
                annotated_image,
                pt1=(xmin, ymin),
                pt2=(xmin + width, ymin + height),
                color=(255, 255, 0),
                thickness=5,
            )
            print(xmin, ymin, width, height)
            face_info = {"top": ymin, "left": xmin, "width": width, "height": height}
            print(
                mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.NOSE_TIP
                )
            )
            point = mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.NOSE_TIP
            )
            center_point = int(point.x * w), int(point.y * h)
            print("center point is ", int(point.x * w), int(point.y * h))
            mp_drawing.draw_detection(annotated_image, detection)
        cv2.namedWindow("facemesh", cv2.WINDOW_NORMAL)
        cv2.imshow("facemesh", annotated_image)
        # cv2.waitKey(0)
        # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
        return face_info, center_point


if __name__ == "__main__":
    file = "/home/whm/workspace/segmentataion/u-2-net-portrait/dataset/demo/7.jpg"
    from face_crop import get_crop_img
    from inference import run
    from lib.utils.oom import free_up_memory

    image = cv2.imread(file)
    face_info, points = get_face_key_point(image)
    img_crop = get_crop_img(image, face_info, points)
    res_img = run(img_crop)
    cv2.imwrite("temp3.jpeg", res_img)
    cv2.namedWindow("img_crop", cv2.WINDOW_NORMAL)
    cv2.imshow("img_crop", res_img)
    cv2.waitKey(0)
    free_up_memory()
