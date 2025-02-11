from flask import Flask, jsonify
import requests
import cv2

app = Flask(__name__)


def webcam():


    # Open the webcam (0 = default webcam)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


@app.route('/call_server')
def call_server():
    
    webcam()
    
    try:
        response = requests.get("http://147.79.70.201:5000/detect")  # Call the Flask server
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run client on a different port














