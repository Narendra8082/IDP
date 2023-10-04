import cv2

# Input video file
input_file = 'walking.mp4'

# Output video file
output_file = 'output_video.avi'

# Open the video file
cap = cv2.VideoCapture(input_file)

# Get the frames per second (fps) and frame size of the input video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Decrease the frame size (resolution) for the output video
output_frame_width = frame_width // 4  # Adjust the width as needed
output_frame_height = frame_height // 4  # Adjust the height as needed

# Create VideoWriter object to save the output with the decreased frame size
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (output_frame_width, output_frame_height))

# Initialize the HOG descriptor for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if we have reached the end of the video
    if not ret:
        break

    # Resize the frame to the desired output frame size
    frame = cv2.resize(frame, (output_frame_width, output_frame_height))

    # Detect humans in the frame
    humans, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)

    # Draw rectangles around detected humans
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Write the frame with bounding boxes to the output video
    out.write(frame)

    # Display the frame with bounding boxes (optional)
    cv2.imshow('Human Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video objects
cap.release()
out.release()
cv2.destroyAllWindows()