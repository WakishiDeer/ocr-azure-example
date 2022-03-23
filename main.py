from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

import time
import cv2

import env

frame = None


def read_image_from_camera():
    capture = cv2.VideoCapture(0)
    global frame
    while True:
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("Press `q` when you want to read.", frame)

    capture.release()
    cv2.destroyAllWindows()
    return frame


'''
Authenticate
Authenticates your credentials and creates a client.
'''
subscription_key = env.my_subscription_key
endpoint = env.my_endpoint
# make client
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

print("===== Read File - local =====")
read_image = read_image_from_camera()
cv2.imwrite("images_folder/read_image.jpg", read_image)
bin_jpeg = open("images_folder/read_image.jpg", "rb")

# if you want to use url of image file, remove initial character of comment
# image_url = "https://www.webcartop.jp/wp-content/uploads/2020/06/TOP-54-680x453.jpg"
# read_response = computervision_client.read(image_url, raw=True)

# Call API with image and raw response (allows you to get the operation location)
read_response = computervision_client.read_in_stream(bin_jpeg, raw=True)

# Get the operation location (URL with ID as last appendage)
read_operation_location = read_response.headers["Operation-Location"]
# Take the ID off and use to get results
operation_id = read_operation_location.split("/")[-1]

# Call the "GET" API and wait for the retrieval of the results
while True:
    read_result = computervision_client.get_read_result(operation_id)
    if read_result.status.lower() not in ['notstarted', 'running']:
        break
    print('Waiting for result...')
    time.sleep(10)

# Print results, line by line
if read_result.status == OperationStatusCodes.succeeded:
    for text_result in read_result.analyze_result.read_results:
        for line in text_result.lines:
            print(line.text)
            print(line.bounding_box)
print()
