import sys
import torch
from hubconf import custom
import supervision as sv
import numpy as np
import cv2
import time
import os

# Load the YOLOv7 model with specific weights for object detection.
model = custom(path_or_model="yolov7-d6.pt", autoshape=True)

# Define the list of video files to process.
video_files = [
    "cam-rear-2024-05-29_14_36_17.mpg",
    "cam-rear-2024-05-29_14_37_01.mpg",
    "cam-rear-2024-05-29_14_38_05.mpg",
    "cam-rear-2024-05-29_14_39_11.mpg",
    "cam-rear-2024-05-29_14_40_16.mpg",
    "cam-rear-2024-05-29_14_41_00.mpg",
    "cam-rear-2024-05-29_14_42_05.mpg",
    "cam-rear-2024-05-29_14_43_11.mpg",
    "cam-rear-2024-05-29_14_44_16.mpg",
    "cam-rear-2024-05-29_14_45_00.mpg",
    "cam-rear-2024-05-29_14_46_05.mpg",
    "cam-rear-2024-05-29_14_47_14.mpg",
    "cam-rear-2024-05-29_14_48_20.mpg"
]

# Function to initialize global variables.
def initialize_globals():
    """
    Initializes all the global variables used in the tracking and detection process.
    """
    global max_persons, max_objects, alert_triggered, object_to_person, missing_times, item_first_detected, owner_assigned_time, entry_line_crossed, detected_entries, polygon_zone, polygon_zone_annotator
    max_persons = 0
    max_objects = 0
    alert_triggered = False
    object_to_person = {}  # Dictionary to map object IDs to their assigned person IDs.
    missing_times = {}  # Dictionary to track the time when objects went missing.
    item_first_detected = {}  # Dictionary to track the first detection time of objects.
    owner_assigned_time = {}  # Dictionary to track the time when ownership was assigned to objects.
    entry_line_crossed = {}  # Dictionary to track objects and persons that have crossed the entry line.
    detected_entries = []  # List to store tuples of detected person and object pairs.
    polygon_zone = None  # Placeholder for the polygon zone.
    polygon_zone_annotator = None  # Placeholder for the polygon zone annotator.

# Initialize the tracker and annotators.
tracker = sv.ByteTrack()  # Initialize the tracker for tracking objects across frames.
box_annotator = sv.BoundingBoxAnnotator()  # Initialize the annotator for bounding boxes.
label_annotator = sv.LabelAnnotator()  # Initialize the annotator for labels.
trace_annotator = sv.TraceAnnotator()  # Initialize the annotator for traces.

# User-defined alert duration in seconds.
alert_duration = 3

# Confidence threshold for filtering detections.
conf_threshold = 0.1

# Define the polygon zone for the rear camera with coordinates.
polygon = np.array([[0, 1200], [450, 350], [1300, 350], [1900, 1200]])

# Define the entry line coordinates for the rear camera.
entry_line = [(1150, 350), (1400, 1200)]

# Function to check if a bounding box crosses the entry line.
def crosses_entry_line(bbox, line):
    """
    Checks if a bounding box crosses the entry line.

    Args:
    bbox (list): Bounding box coordinates [x1, y1, x2, y2].
    line (list): Entry line coordinates [(x1, y1), (x2, y2)].

    Returns:
    bool: True if the bounding box crosses the entry line, False otherwise.
    """
    x1, y1, x2, y2 = bbox
    line_x1, line_y1, line_x2, line_y2 = line[0][0], line[0][1], line[1][0], line[1][1]
    return ((x1 <= line_x1 <= x2 or x1 <= line_x2 <= x2) and (y1 <= line_y1 <= y2 or y1 <= line_y2 <= y2))

# Function to assign objects to the nearest person permanently.
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
            obj_center = [(obj[0] + obj[2]) / 2, (obj[1] + obj[3]) / 2]
            min_distance = float('inf')
            assigned_person = None
            for j, person in enumerate(person_detections.xyxy):
                person_center = [(person[0] + person[2]) / 2, (person[1] + person[3]) / 2]
                distance = np.linalg.norm(np.array(obj_center) - np.array(person_center))
                if distance < min_distance:
                    min_distance = distance
                    assigned_person = person_detections.tracker_id[j]
            if assigned_person is not None:
                object_to_person[obj_id] = assigned_person
                owner_assigned_time[obj_id] = time.time()
    return object_to_person

# Define the callback function for processing frames.
def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
    """
    Processes each frame to detect objects and persons, assign ownership, and generate alerts.

    Args:
    frame (np.ndarray): The current video frame.
    frame_index (int): The index of the current frame.

    Returns:
    np.ndarray: The annotated frame.
    """
    global max_persons, max_objects, alert_triggered, polygon_zone, polygon_zone_annotator, object_to_person, missing_times, item_first_detected, owner_assigned_time, entry_line_crossed, detected_entries

    # Initialize the polygon zone with frame dimensions.
    if polygon_zone is None:
        frame_height, frame_width = frame.shape[:2]
        polygon_zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(frame_width, frame_height))
        polygon_zone_annotator = sv.PolygonZoneAnnotator(zone=polygon_zone, color=sv.Color.YELLOW, thickness=2)

    # Run object detection on the current frame.
    results = model(frame)
    detections = sv.Detections.from_yolov7(results)
    detections = detections[detections.confidence > conf_threshold]  # Applying confidence threshold.

    # Filter detections based on relative area and exclude "tv" class.
    image_area = frame.shape[0] * frame.shape[1]
    detections = detections[(detections.area / image_area) < 0.1]  # Adjusting the threshold to filter large detections.
    tv_class_id = results.names.index("tv")  # Get the class ID for "tv".
    detections = detections[detections.class_id != tv_class_id]  # Filter out detections of class "tv".

    # Update the tracker with the filtered detections.
    detections = tracker.update_with_detections(detections)

    # Apply the polygon zone mask to the detections.
    polygon_mask = polygon_zone.trigger(detections=detections)
    detections_in_zone = sv.Detections(
        xyxy=detections.xyxy[polygon_mask],
        confidence=detections.confidence[polygon_mask],
        class_id=detections.class_id[polygon_mask],
        tracker_id=detections.tracker_id[polygon_mask]
    )

    # Get class ID for "person".
    person_class_id = results.names.index("person")

    # Filter detections to separate persons and other objects.
    person_mask = detections_in_zone.class_id == person_class_id
    object_mask = detections_in_zone.class_id != person_class_id

    person_detections = sv.Detections(
        xyxy=detections_in_zone.xyxy[person_mask],
        confidence=detections_in_zone.confidence[person_mask],
        class_id=detections_in_zone.class_id[person_mask],
        tracker_id=detections_in_zone.tracker_id[person_mask]
    )

    object_detections = sv.Detections(
        xyxy=detections_in_zone.xyxy[object_mask],
        confidence=detections_in_zone.confidence[object_mask],
        class_id=detections_in_zone.class_id[object_mask],
        tracker_id=detections_in_zone.tracker_id[object_mask]
    )

    # Record the initial detection time for each item.
    current_time = time.time()
    for obj_id in object_detections.tracker_id:
        if obj_id not in item_first_detected:
            item_first_detected[obj_id] = current_time

    # Track if objects and persons cross the entry line.
    for i, obj in enumerate(object_detections.xyxy):
        obj_id = object_detections.tracker_id[i]
        if crosses_entry_line(obj, entry_line):
            entry_line_crossed[obj_id] = 'object'

    for j, person in enumerate(person_detections.xyxy):
        person_id = person_detections.tracker_id[j]
        if crosses_entry_line(person, entry_line):
            entry_line_crossed[person_id] = 'person'

    # Assign ownership based on entry line crossing.
    for obj_id, obj_crossed in entry_line_crossed.items():
        if obj_crossed == 'object':
            for person_id, person_crossed in entry_line_crossed.items():
                if person_crossed == 'person':
                    detected_entries.append((person_id, obj_id))
                    break

    # Assign ownership if both person and object are detected crossing the entry line.
    for person_id, obj_id in detected_entries:
        if person_id in person_detections.tracker_id and obj_id in object_detections.tracker_id:
            object_to_person[obj_id] = person_id
            owner_assigned_time[obj_id] = time.time()

    # Assign possession based on proximity if not assigned by entry line.
    object_to_person = assign_possession(person_detections, object_detections, object_to_person)

    # Count the number of persons and objects in the current frame within the zone.
    num_persons = sum(1 for class_id in detections_in_zone.class_id if results.names[class_id] == "person")
    num_objects = len(detections_in_zone.class_id) - num_persons

    # Update the maximum counts.
    max_persons = max(max_persons, num_persons)
    max_objects = max(max_objects, num_objects)

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

    # Create labels for the detections.
    labels = [
        f"#{tracker_id} {results.names[class_id]} {confidence:0.2f}"
        if class_id == person_class_id else 
        f"#{tracker_id} {results.names[class_id]} {confidence:0.2f} | Owner: {object_to_person.get(tracker_id, 'None')} | Status: {'Present' if object_to_person.get(tracker_id) in persons_present else 'Missing'}"
        for class_id, confidence, tracker_id
        in zip(detections_in_zone.class_id, detections_in_zone.confidence, detections_in_zone.tracker_id)
    ]

    # Annotate the frame with bounding boxes, labels, and traces.
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections_in_zone)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections_in_zone, labels=labels)
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections_in_zone)

    # Annotate the frame with the polygon zone and count of detected objects within the zone.
    annotated_frame = polygon_zone_annotator.annotate(scene=annotated_frame, label=f"People: {num_persons} | Objects: {num_objects}")

    # Add the counter information to the frame using OpenCV.
    counter_info = f"Max People: {max_persons} | Max Objects: {max_objects}"
    position = (10, annotated_frame.shape[0] - 10)
    font_scale = 1.0
    color = (255, 255, 255)
    thickness = 2
    cv2.putText(annotated_frame, counter_info, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    # Add alert message if triggered.
    if alert_triggered:
        alert_text = "ALERT: Item left behind!"
        alert_position = (10, 50)
        alert_color = (0, 0, 255)  # Red color for alert
        cv2.putText(annotated_frame, alert_text, alert_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, alert_color, thickness)

    # Draw the entry line on the frame and add a label.
    cv2.line(annotated_frame, entry_line[0], entry_line[1], (0, 255, 255), 2)  # Yellow color for entry line
    mid_point = ((entry_line[0][0] + entry_line[1][0]) // 2, (entry_line[0][1] + entry_line[1][1]) // 2)
    cv2.putText(annotated_frame, "Entry Line", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    return annotated_frame

# Process each video with the defined callback.
for video_file in video_files:
    initialize_globals()  # Reset global variables for each video
    target_file = f"processed_rear_camera_{video_file}"
    sv.process_video(source_path=video_file, target_path=target_file, callback=callback)
