import cv2

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def find_and_blur(bw, color): 
    # detect al faces
    faces = cascade.detectMultiScale(bw, 1.1, 4)
    # get the locations of the faces
    for (x, y, w, h) in faces:
        # select the areas where the face was found
        roi_color = color[y:y+h, x:x+w]
        # blur the colored image
        blur = cv2.GaussianBlur(roi_color, (101,101), 0)
        # Insert ROI back into image
        color[y:y+h, x:x+w] = blur            
    
    # return the blurred image
    return color

# turn camera on
video_capture = cv2.VideoCapture(0)
if video_capture.isOpened(): 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width  = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # float value of original video
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) #float value of original video
    fps = video_capture.get(cv2.CAP_PROP_FPS) #fps of original video
    out = cv2.VideoWriter("output_haar.mp4", fourcc, 10, (int(width),int(height)))

while True:         
    # get last recorded frame
    _, frame = video_capture.read()
    # transform color -> grayscale
    bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect the face and blur it
    blur = find_and_blur(bw, frame)
    #store output
    out.write(blur)
    # display output
    cv2.imshow('Video', blur)
    # break if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# turn camera off        
video_capture.release()
out.release()
# close camera  window
cv2.destroyAllWindows()