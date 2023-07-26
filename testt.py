import cv2
import numpy as np

def calculate_angle(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            return angle

def main():
    # Open a video capture object
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, threshold1=30, threshold2=100)

        # Calculate the angle
        angle = calculate_angle(edges)
        if angle is not None:
            # Display the bending angle of the sheet
            cv2.putText(frame, "Bent angle: {:.2f} degrees".format(angle), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame and the edges
        cv2.imshow("Metal Sheet", frame)
        cv2.imshow("Edges", edges)
        # Check for key press


        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture object and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


