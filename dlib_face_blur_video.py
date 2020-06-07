import numpy as np
import cv2
import dlib
import sys
import argparse


# handle command line arguments
ap = argparse.ArgumentParser()
#ap.add_argument('--image', required=True, help='path to image file')
ap.add_argument('--video', default=0, help='path to video file or camera id')
args = ap.parse_args()

video_cap = cv2.VideoCapture(args.video)
detector = dlib.get_frontal_face_detector()

blurred = True

while True:
	#cap frame-by-frame
	ret, frame = video_cap.read()
	#create temp img
	temp_img = frame.copy()
	mask_shape = (frame.shape[0], frame.shape[1], 1)
	mask = np.full(mask_shape, 0, dtype=np.uint8)

	if (ret):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	

		rects = detector(gray,0)

		for rect in rects:
			x = rect.left()
			y = rect.top()
			x1 = rect.right()
			y1 = rect.bottom()
			h = rect.height()
			w = rect.width()

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

