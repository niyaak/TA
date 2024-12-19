import cv2
import numpy as np

# Load the ENet model
model = cv2.dnn.readNet('D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/enet-model.net')

# Normalization factor
norm_factor = 1 / 255.0

# Load the video
cap = cv2.VideoCapture('D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/demo1.mp4')

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video properties (width, height, FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object to save the output video
out = cv2.VideoWriter('D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/output/soutput_video.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (frame_width, frame_height))

# Assuming the model expects input size of 848x480
input_size = (848, 480)

while True:
    # Read frame from video
    ret, frame = cap.read()

    # If the frame is read correctly, ret is True
    if not ret:
        print("End of video or error reading frame.")
        break

    # Convert frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Image preprocessing without resize
    proc_img = cv2.dnn.blobFromImage(image=img, scalefactor=norm_factor, size=input_size, mean=0, swapRB=True, crop=False)

    # Model evaluation
    model.setInput(proc_img)
    eval = model.forward()[0]

    # Pixels class prediction (argmax to get the predicted class for each pixel)
    inds = np.argmax(eval, axis=0)

    # Road binarization (class 1 for road)
    road_bin = inds == 1
    road_bin = np.array(road_bin, dtype=np.float32)

    # Gradient calculation for edge detection
    gx, gy = np.gradient(road_bin)
    edge = gy * gy + gx * gx  # Gradient magnitude
    edge[edge != 0.0] = 255.0  # Elements with gradient different from 0
    edge = np.array(edge, dtype=np.float32)

    # Apply slight Gaussian Blur to smooth edges (but less than before)
    edge = cv2.GaussianBlur(edge, (5, 5), 0)

    # Apply Median Blur with a smaller kernel
    road_bin = cv2.medianBlur(road_bin.astype(np.uint8), 3)

    # Dilation and Erosion with fewer iterations
    kernel = np.ones((3, 3), np.uint8)
    road_bin = cv2.dilate(road_bin, kernel, iterations=1)
    road_bin = cv2.erode(road_bin, kernel, iterations=1)

    # Create an output image (overlay road and edge detection)
    db = 50  # Segmentation intensity
    out_img = frame.copy()

    for ri in range(frame.shape[0]):
        for ci in range(frame.shape[1]):
            # Edges
            if edge[ri, ci] > 0:
                out_img[ri, ci] = [0, 0, 120]  # Highlight edges in blue

            # Road
            if road_bin[ri, ci] > 0:
                if frame[ri, ci, 2] + db > 255:
                    out_img[ri, ci, 2] = 255  # Adjust the red channel
                else:
                    out_img[ri, ci, 2] += db

    # Write the frame to the output video
    out.write(out_img)

    # Display the processed frame
    cv2.imshow('Processed Video', out_img)

    # Wait for 1 ms before the next frame; press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
