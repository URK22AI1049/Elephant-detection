import cv2
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
from imageai.Detection import ObjectDetection
from sklearn.externals import joblib

clf = joblib.load('data.pkl')

cap = cv2.VideoCapture(0)

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("resnet50_coco_best_v2.0.1.h5")
detector.loadModel()

while True:
    ret, frame = cap.read()

    gray_frame = rgb2gray(frame)

    resized_frame = resize(gray_frame, (128, 128))

    hog_features = hog(resized_frame, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2')

    hog_features = hog_features.reshape(1, -1)

    prediction = clf.predict(hog_features)

    if prediction == 1:
        detections = detector.detectObjectsFromImage(input_image=frame, output_image_path="image_with_detections.jpg", minimum_percentage_probability=50)

        for detection in detections:
            if detection["name"] == "elephant":
                (x1, y1, x2, y2) = detection["box_points"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Elephant", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
