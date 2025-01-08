# YOLOv7-based Object Detection System to Detect Left Behind Items in Autonomous Public Transport

## Installation
- Clone the YOLOv7 repository:
```
git clone https://github.com/WongKinYiu/yolov7
```
- Use the requirements.txt file in the YOLOv7 repository to install the required libraries:
```
pip install -r requirements.txt
```
- Place the Python script 'left_behind_inside_camera.py' and 'left_behind_rear_camera.py' inside the repository directory alongside 'hubconf.py'.
- Place the 'yolov7-d6.pt' weight file in the same directory as the script.
- Place your video files in the same directory as the script.

#### Usage
To run the script, simply execute the following command:
```
python left_behind_inside_camera.py
python left_behind_rear_camera.py
```

## Overview
This project utilizes computer vision techniques to identify and track items brought by passengers in an autonomous public transport vehicle. The system associates items with their respective owners as they enter the vehicle and triggers an alert if any item is left behind when the passenger exits. This enhances security and convenience for passengers by ensuring no items are left behind.

## Object Detection and Tracking
We have a camera feed from the inside of the vehicle and need to reliably detect people and items in the feed.
- YOLOv7 (You Only Look Once): A single-stage object detector for object detection, pretrained on the COCO dataset with 80 different labels.
- ByteTrack: Used for assigning unique tracking IDs to people and objects to maintain consistency across frames.

## Reducing Object Detection Noise
The real-time video feed from the vehicle is not of the highest bitrate. Some items are always present in the vehicle and do not contribute to the problem we are trying to solve. Irrelevant detections are filtered out based on class IDs, and unwanted large detections are filtered out based on the relative size of the bounding box compared to the frame size. A polygon zone is created to filter out all detections outside the specified zone, counting only objects and people within the zone. This reduces noise from irrelevant objects.


## Item-Person Association
We want to correctly pair the item with its owner, and for that, we are using two methods: entry line ownership assignment and proximity-based assignment. Below is an explanation of each method:

### Entry Line Based Assignment
The entry line ownership assignment is used to pair an item with its owner when both cross the entry line simultaneously or in quick succession.

- The entry line is defined by two points in the frame:
```
entry_line = [(350, 1200), (600, 150)]
```
- The crosses_entry_line function checks if a bounding box (person or object) crosses the predefined entry line. This is done by checking if any part of the bounding box intersects the entry line.
```
def crosses_entry_line(bbox, line):
    x1, y1, x2, y2 = bbox
    line_x1, line_y1, line_x2, line_y2 = line[0][0], line[0][1], line[1][0], line[1][1]
    return ((x1 <= line_x1 <= x2 or x1 <= line_x2 <= x2) and (y1 <= line_y1 <= y2 or y1 <= line_y2 <= y2))
```
- During each frame, the code checks if any person or object crosses the entry line. If a bounding box crosses the line, it is recorded in the entry_line_crossed dictionary:
```
for i, obj in enumerate(object_detections.xyxy):
    obj_id = object_detections.tracker_id[i]
    if crosses_entry_line(obj, entry_line):
        entry_line_crossed[obj_id] = 'object'

for j, person in enumerate(person_detections.xyxy):
    person_id = person_detections.tracker_id[j]
    if crosses_entry_line(person, entry_line):
        entry_line_crossed[person_id] = 'person'
```
- If both a person and an object are detected crossing the entry line within a certain time frame, their IDs are paired and stored in detected_entries. Once an object is assigned to a person using the entry line method, the assignment is permanent for the duration of the tracking session. This ensures that their association remains intact even if they move around.
```
for obj_id, obj_crossed in entry_line_crossed.items():
    if obj_crossed == 'object':
        for person_id, person_crossed in entry_line_crossed.items():
            if person_crossed == 'person':
                detected_entries.append((person_id, obj_id))
                break

for person_id, obj_id in detected_entries:
    if person_id in person_detections.tracker_id and obj_id in object_detections.tracker_id:
        object_to_person[obj_id] = person_id
        owner_assigned_time[obj_id] = time.time()
```

### Proximity-Based Assignment
The proximity-based assignment function is used when the entry line logic does not capture the detection of both a person and their associated object simultaneously. This function ensures that objects are assigned to the nearest detected person based on the distance between their bounding boxes.

```
def assign_possession(person_detections, object_detections, object_to_person):
    """
    Assigns objects to the nearest detected person permanently.

    Args:
    person_detections (sv.Detections): Detections of persons.
    object_detections (sv.Detections): Detections of objects.
    object_to_person (dict): Dictionary mapping object IDs to person IDs.

    Returns:
    dict: Updated dictionary mapping object IDs to person IDs.
    """
    for i, obj in enumerate(object_detections.xyxy):
        obj_id = object_detections.tracker_id[i]
        if obj_id not in object_to_person:
            # Calculate the center point of the object
            obj_center = [(obj[0] + obj[2]) / 2, (obj[1] + obj[3]) / 2]
            min_distance = float('inf')
            assigned_person = None
            for j, person in enumerate(person_detections.xyxy):
                # Calculate the center point of the person
                person_center = [(person[0] + person[2]) / 2, (person[1] + person[3]) / 2]
                # Compute the Euclidean distance between the object center and the person center
                distance = np.linalg.norm(np.array(obj_center) - np.array(person_center))
                # If this distance is the smallest found so far, update the assigned person
                if distance < min_distance:
                    min_distance = distance
                    assigned_person = person_detections.tracker_id[j]
            if assigned_person is not None:
                # Assign the object to the nearest person
                object_to_person[obj_id] = assigned_person
                # Record the time when ownership was assigned
                owner_assigned_time[obj_id] = time.time()
    return object_to_person
```

- The function iterates over each detected object (object_detections.xyxy). For each object, it retrieves its unique identifier (obj_id).
- The function checks if the object is already assigned to a person. If not, it proceeds with the assignment.
- The center of the object’s bounding box is calculated using its coordinates.
- A variable min_distance is initialized to infinity to keep track of the smallest distance found.
- The function iterates over each detected person (person_detections.xyxy). For each person, it calculates the center of their bounding box.
- The Euclidean distance between the center of the object and the center of the person is computed using np.linalg.norm.
- If the calculated distance is smaller than the current min_distance, the function updates min_distance and sets assigned_person to the current person’s ID.
- After finding the nearest person, the function assigns the object to this person and records the assignment time.
- Similar to the entry line method, once an object is assigned to a person using the proximity method, the assignment is permanent for the duration of the tracking session. This ensures that their association remains intact even if they move around.
Alert Logic

#### Example Scenario:
Entry Line Missed: A person crosses the entry line with an object, but the object is not detected. Later, another person crosses, and only their object is detected.
Proximity Assignment: The system will use the proximity-based assignment to link objects to the nearest detected person, ensuring ownership is accurately established even when entry line detection fails.

## Alert Logic
The alert logic in the code is designed to trigger an alert if an object is left behind by its assigned owner.
```
# Check if the assigned person for any object is no longer present.
alert_triggered = False
persons_present = set(detections_in_zone.tracker_id[detections_in_zone.class_id == person_class_id])
for obj_id, person_id in object_to_person.items():
    if person_id not in persons_present:
        if obj_id in owner_assigned_time:
            if obj_id not in missing_times:
                missing_times[obj_id] = current_time
            elif current_time - missing_times[obj_id] >= alert_duration:
                alert_triggered = True
                break
    else:
        if obj_id in missing_times:
            del missing_times[obj_id]

# Add alert message if triggered.
if alert_triggered:
    alert_text = "ALERT: Item left behind!"
    alert_position = (10, 50)
    alert_color = (0, 0, 255)  # Red color for alert
    cv2.putText(annotated_frame, alert_text, alert_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, alert_color, thickness)

```
- 'alert_duration' variable defines the duration (in seconds) after which an alert is triggered if an object is detected without its assigned owner.
- During each frame, the code maintains a list of 'persons_present', which contains the tracker IDs of all persons detected in the current frame within the polygon zone.
- The code iterates through the dictionary 'object_to_person', which maps object IDs to their assigned person IDs.
-For each object, it checks if the assigned person is still present in the current frame.
- If the assigned person is not detected in the current frame ('person_id not in persons_present'):
  - The code checks if the object has been missing for a significant amount of time.
  - If the object was not previously recorded as missing, it records the current time in missing_times for that object.
  - If the object was already missing, the code calculates the duration it has been missing by comparing the current time with the recorded missing time.
  - If the object has been missing for longer than alert_duration, an alert is triggered.
- If an object has been left behind (i.e., its assigned person has been missing for longer than 'alert_duration'), the 'alert_triggered flag' is set to True. The alert message "ALERT: Item left behind!" is added to the annotated frame.
- If the assigned person is present in the current frame, any previously recorded missing time for that object is removed from missing_times.

## Credits
https://github.com/WongKinYiu/yolov7

https://github.com/ifzhang/ByteTrack

https://github.com/roboflow/supervision



