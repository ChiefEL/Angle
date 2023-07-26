import cv2
import numpy as np
import matplotlib.pyplot as plt


# extract blob with largest area
def get_best_blob(blobs):
    best_blob = None
    
    best_size = 0
    for i, blob in enumerate(blobs):
        rot_rect = cv2.minAreaRect(blob)
        (cx, cy), (sx, sy), angle = rot_rect
        if sx * sy > best_size:
            best_blob = rot_rect
            best_size = sx * sy

    return best_blob


def draw_blob_rect(frame, blob, color):
    box = cv2.boxPoints(blob)
    box = np.int0(box)
    frame = cv2.drawContours(frame, [box], 0, color, 1)

    return frame


def process():
    image = cv2.imread("wire.png")

    # Change to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce size for speed
    gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

    # Extract wire region
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # morphology kernel
    kernel = np.array((
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ), dtype="int")

    # Get y-edges
    dy = cv2.Sobel(thresh, cv2.CV_32F, 0, 1, ksize=21)

    # remove negative values
    dy = dy * dy

    # Normalize it to 255
    cv2.normalize(dy, dy, norm_type=cv2.NORM_MINMAX, alpha=255)
    dy = dy.astype('uint8')

    # Extract relevant information
    # Stronger edge is the original part
    # Weaker edge is the bended part
    _, strong = cv2.threshold(dy, 0.95 * 255, 255, cv2.THRESH_BINARY)
    _, mid = cv2.threshold(dy, 0.5 * 255, 255, cv2.THRESH_BINARY)

    # Morphological closing to remove holes
    strong_temp = cv2.dilate(strong, kernel, iterations=5)
    strong = cv2.erode(strong_temp, kernel, iterations=5)

    # remove the strong part from the mid
    mid = cv2.subtract(mid, strong_temp)

    # Morphological closing to remove holes
    mid_temp = cv2.dilate(mid, kernel, iterations=5)
    mid = cv2.erode(mid_temp, kernel, iterations=5)

    # find the blobs for each bin image
    _, strong_blobs, _ = cv2.findContours(strong, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, mid_blobs, _ = cv2.findContours(mid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get the blob with the largest area
    best_strong = get_best_blob(strong_blobs)
    best_mid = get_best_blob(mid_blobs)

    # print the angle
    print("strong_angle", 90 + best_strong[2])
    print("mid_angle", 90 + best_mid[2])

    # Draw the segmented Box region
    display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    display_frame = draw_blob_rect(display_frame, best_strong, (0, 0, 255))
    display_frame = draw_blob_rect(display_frame, best_mid, (0, 255, 255))

    # draw result
    cv2.imshow("display_frame", display_frame)
    cv2.imshow("mid", mid)
    cv2.imshow("strong", strong)
    cv2.imshow("image", image)
    cv2.imshow("dy", dy)
    cv2.waitKey(0)


if __name__ == '__main__':
    process()