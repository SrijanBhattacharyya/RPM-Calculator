import cv2
import numpy as np

def calculate_rpm(frames, centroids):
    displacements = np.diff(centroids)
    periods = displacements / (frames / 2)
    rpm = 60 / periods
    return rpm

def main():
    # Initialize the camera
    camera = cv2.VideoCapture(0)

    # Initialize the frames and centroids
    frames = []
    centroids = []

    # Capture frames from the camera
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Find the centroid of the rotating object
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            centroid = np.mean(contours[0], axis=0)

        # Add the frame and centroid to the lists
        frames.append(frame)
        centroids.append(centroid)

    # Calculate the RPM
    rpm = calculate_rpm(frames, centroids)

    # Print the RPM
    print("RPM:", rpm)

if __name__ == "__main__":
    main()
      
