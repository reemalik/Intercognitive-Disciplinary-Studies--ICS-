import cv2
from gui_buttons import Buttons


#buttons
button = Buttons()
button.add_button("person", 20,20)
button.add_button("cell phone", 20, 50)
button.add_button("clock", 20, 80)
button.add_button("remote", 20, 110)
button.add_button("scissors", 20, 140)

colors= button.colors


#OpenCV DNN loading the neural network
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights" , "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams (size=(320,320), scale=1/255)


#load class lists
classes=[]
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name= class_name.strip()
        classes.append(class_name)

print("Objects list")
print(classes)



#initialising webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1288)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#Full HD




#Mouse Events
def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
            button.button_click(x,y)



#create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)




while True:
    # getting frames from webcam
    ret, frame = cap.read()


    # get active buttons
    active_buttons= button.active_buttons_list()
    print("Active btns" , active_buttons)


    #Object detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        color = colors[class_id]

        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 0, 50), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (200, 0, 50), 3)

    #Display Buttons
    button.display_buttons(frame)


    # print("class ids", class_ids)
    # print("scores", scores)
    # print("bboxes", bboxes)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()