import numpy as np
import cv2
import dlib
import argparse


# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('--weights', default='./mmod_human_face_detector.dat',
                help='path to weights file')
args = ap.parse_args()

video_cap = cv2.VideoCapture(0)
cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)

blurred = False
framed = False

while True:
	#cap frame-by-frame
	ret, frame = video_cap.read()
	#create temp img
	temp_img = frame.copy()
	mask_shape = (frame.shape[0], frame.shape[1], 1)
	mask = np.full(mask_shape, 0, dtype=np.uint8)

	if (ret):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces_cnn = cnn_face_detector(gray,0)

		for face in faces_cnn:
			x = face.rect.left()
			y = face.rect.top()
			x1 = face.rect.right()
			y1 = face.rect.bottom()
			h = face.rect.height()
			w = face.rect.width()

			#frame blur
			temp_img[y:y1, x:x1] = cv2.blur(temp_img[y:y1, x:x1], (25,25))

			#creat circle mask
			cv2.circle(mask, (int((x + x + w)/2), int((y + y + h)/2)), int(h/2),(255), -1)

		#apply the mask and combine
		mask_inv = cv2.bitwise_not(mask)
		img_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
		img_fg = cv2.bitwise_and(temp_img, temp_img, mask=mask)
		combined = cv2.add(img_bg, img_fg)

		#display the frame result
		cv2.imshow('video Feed', combined)

	ch = 0xFF & cv2.waitKey(1)

	#quit
	if ch == ord("q"):
		break

#done all
video_cap.release()
cv2.destroyAllWindows()

