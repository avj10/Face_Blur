import numpy as np
import cv2
import dlib

video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

if video_capture.isOpened(): 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width  = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # float value of original video
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) #float value of original video
    out = cv2.VideoWriter("output_circle.mp4", fourcc, 10, (int(width),int(height)))

while True:
	#cap frame-by-frame
	ret, frame = video_capture.read()
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
			temp_img[y:y1, x:x1] = cv2.blur(temp_img[y:y1, x:x1], (50,50))

			#creat circle mask
			cv2.circle(mask, (int((x + x + w)/2), int((y + y + h)/2)), int(h/2),(255), -1)

		#apply the mask and combine
		mask_inv = cv2.bitwise_not(mask)
		img_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
		img_fg = cv2.bitwise_and(temp_img, temp_img, mask=mask)
		combined = cv2.add(img_bg, img_fg)

		#save video
		out.write(combined)
		#display the frame result
		cv2.imshow('video Feed', combined)

	ch = 0xFF & cv2.waitKey(1)

	#quit
	if ch == ord("q"):
		break

#done all
video_capture.release()
out.release()
cv2.destroyAllWindows()

