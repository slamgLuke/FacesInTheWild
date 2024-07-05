import cv2
import dlib
import numpy as np
from imutils import face_utils
import os
import glob

detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def detect_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use Dlib detector
    rects = detector_dlib(gray, 1)

    # No faces detected
    if len(rects) == 0:
        return None
    else:
        return rects[0]


def align_and_crop_face(image, desired_size=(64, 64), face_size=(150, 150), name=""):
    rect = detect_faces(image)

    if rect is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # Compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # Compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # Compute center (x, y)-coordinates (i.e., the median point) between the two eyes in the input image
        eyesCenter = (
            int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
            int((leftEyeCenter[1] + rightEyeCenter[1]) // 2),
        )
        # print(eyesCenter)

        # Grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(center=eyesCenter, angle=angle, scale=1)

        # Apply the affine transformation
        (h, w) = image.shape[:2]
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        # Calculate the coordinates of the face bounding box
        leftX = max(0, eyesCenter[0] - face_size[0] // 2)
        rightX = min(w, eyesCenter[0] + face_size[0] // 2)
        topY = max(0, eyesCenter[1] - face_size[1] // 2)
        bottomY = min(h, eyesCenter[1] + face_size[1] // 2)

        # Crop the face region
        face = rotated[topY:bottomY, leftX:rightX]

        # Resize the face region to the desired size
        face = cv2.resize(face, desired_size, interpolation=cv2.INTER_CUBIC)
        return face

    else:
        print("No face detected for image: ", name, " - Using default crop")
        # Asume face is centered
        (h, w) = image.shape[:2]
        leftX = w // 4
        rightX = 3 * w // 4
        topY = h // 4
        bottomY = 3 * h // 4
        face = image[topY:bottomY, leftX:rightX]
        face = cv2.resize(face, desired_size, interpolation=cv2.INTER_CUBIC)
        return face


for image_path in glob.glob("images/*.png"):
    img = cv2.imread(image_path)
    face = align_and_crop_face(img, name=str(image_path))
    if face is not None:
        cv2.imwrite("cropped_faces/" + os.path.basename(image_path), face)
    else:
        print(f"No face detected in {image_path}")
