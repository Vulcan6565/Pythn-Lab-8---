from pathlib import Path

import cv2
import numpy as np


FOLDER = Path(__file__).resolve().parent
CAMERA_NUMBER = 0
THRESHOLD = 110
MIN_AREA = 300
MISSED_LIMIT = 5


def resize_image(show_window=True):
    filename = FOLDER / "images" / "variant-6.png"
    image = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)

    if image is None:
        print("Cannot open images/variant-6.png")
        return

    height, width = image.shape[:2]
    result = cv2.resize(image, (width * 2, height * 2),
                        interpolation=cv2.INTER_LINEAR)

    output_folder = FOLDER / "results"
    output_folder.mkdir(exist_ok=True)
    output_file = output_folder / "variant-6-x2.png"

    if not cv2.imwrite(str(output_file), result):
        print("Cannot save the enlarged image.")
        return

    print("Original size:", width, "x", height)
    print("New size:", width * 2, "x", height * 2)
    print("Saved to:", output_file)

    if show_window:
        cv2.imshow("Original image", image)
        cv2.imshow("Image enlarged 2 times", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def find_marker(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)


    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    best_rectangle = None
    best_area = MIN_AREA
    frame_area = frame.shape[0] * frame.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        ratio = width / height

        if best_area < area < frame_area / 2 and 0.6 < ratio < 1.6:
            best_area = area
            best_rectangle = (x, y, width, height)

    return best_rectangle


def count_entry(center_x, frame_width, last_side, left_count, right_count):
    if center_x < frame_width // 2:
        current_side = "left"
    else:
        current_side = "right"

    if current_side != last_side:
        if current_side == "left":
            left_count += 1
        else:
            right_count += 1

    return current_side, left_count, right_count


def add_fly(frame, fly, center_x, center_y):
    fly_height, fly_width = fly.shape[:2]
    left = center_x - fly_width // 2
    top = center_y - fly_height // 2

    x1 = max(0, left)
    y1 = max(0, top)
    x2 = min(frame.shape[1], left + fly_width)
    y2 = min(frame.shape[0], top + fly_height)

    if x1 >= x2 or y1 >= y2:
        return

    fly_part = fly[y1 - top:y2 - top, x1 - left:x2 - left]
    alpha = fly_part[:, :, 3] / 255.0
    for channel in range(3):
        frame[y1:y2, x1:x2, channel] = (
            fly_part[:, :, channel] * alpha
            + frame[y1:y2, x1:x2, channel] * (1 - alpha)
        )


def track_marker(show_counters=False, show_fly=False):
    fly = None
    if show_fly:
        fly = cv2.imread(str(FOLDER / "fly64.png"), cv2.IMREAD_UNCHANGED)
        if fly is None or fly.ndim != 3 or fly.shape[2] != 4:
            print("Cannot open fly64.png with its transparency channel.")
            return

    camera = cv2.VideoCapture(CAMERA_NUMBER)
    if not camera.isOpened():
        print("Cannot open the camera. Check permissions and CAMERA_NUMBER.")
        camera.release()
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    left_count = 0
    right_count = 0
    last_side = None
    missed_frames = 0

    print("Press Q or Esc in the video window to stop. R resets the counts.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Cannot read a camera frame.")
                break

            height, width = frame.shape[:2]
            marker = find_marker(frame)

            if marker is not None:
                x, y, marker_width, marker_height = marker
                center_x = x + marker_width // 2
                center_y = y + marker_height // 2
                missed_frames = 0

                if show_counters:
                    last_side, left_count, right_count = count_entry(
                        center_x, width, last_side, left_count, right_count
                    )

                cv2.rectangle(frame, (x, y),
                              (x + marker_width, y + marker_height),
                              (0, 255, 0), 2)

                if show_fly:
                    add_fly(frame, fly, center_x, center_y)
                else:
                    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            else:
                missed_frames += 1
                if missed_frames >= MISSED_LIMIT:
                    last_side = None
                cv2.putText(frame, "Marker not found", (10, height - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if show_counters:
                cv2.line(frame, (width // 2, 0), (width // 2, height - 1),
                         (255, 0, 0), 2)
                cv2.rectangle(frame, (0, 0), (width - 1, 45), (35, 35, 35), -1)
                cv2.putText(frame, "Left: " + str(left_count), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Right: " + str(right_count),
                            (width // 2 + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, "Q / Esc: exit   R: reset", (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 0), 1)
            cv2.imshow("Lab 8 - variant 6", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("r"):
                left_count = 0
                right_count = 0
                last_side = None
                missed_frames = 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    while True:
        print("\nLab 8. OpenCV. Variant 6")
        print("1 - Enlarge the image 2 times")
        print("2 - Track the printed marker")
        print("3 - Track the marker and count entries into each half")
        print("4 - Counters and the fly overlay (extra task)")
        print("0 - Exit")
        choice = input("Choose a task: ").strip()

        if choice == "1":
            resize_image()
        elif choice == "2":
            track_marker()
        elif choice == "3":
            track_marker(show_counters=True)
        elif choice == "4":
            track_marker(show_counters=True, show_fly=True)
        elif choice == "0":
            break
        else:
            print("Please enter a number from 0 to 4.")


if __name__ == "__main__":
    main()
