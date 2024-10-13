import face_recognition
import cv2 as cv
import numpy as np
import pickle
import os
import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import Image, ImageTk
import threading
import queue

# Global variables
face_data = {}
root = None
tree = None
recognition_running = False
frame_queue = queue.Queue(maxsize=1)
new_face_data = queue.Queue()
viewer_thread = None
cap = None

script_dir = os.path.dirname(os.path.abspath(__file__))


def initialize_face_data():
    global face_data
    encodings_dir = os.path.join(script_dir, 'face_encodings')
    images_dir = os.path.join(script_dir, 'face_images')
    
    if not os.path.exists(encodings_dir):
        os.makedirs(encodings_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    face_data_path = os.path.join(encodings_dir, 'face_data.pkl')
    
    try:
        with open(face_data_path, 'rb') as f:
            face_data = pickle.load(f)
        for key in ['encodings', 'names', 'image_paths']:
            if key not in face_data:
                face_data[key] = []
        # Ensure all lists have the same length
        min_length = min(len(face_data['encodings']), len(face_data['names']), len(face_data['image_paths']))
        face_data['encodings'] = face_data['encodings'][:min_length]
        face_data['names'] = face_data['names'][:min_length]
        face_data['image_paths'] = face_data['image_paths'][:min_length]
        print(f"Loaded face data: {len(face_data['names'])} faces")
    except FileNotFoundError:
        face_data = {'encodings': [], 'names': [], 'image_paths': []}
        print("No existing face data found. Starting fresh.")
    print(f"Face data keys: {list(face_data.keys())}")

def save_face_data():
    face_data_path = os.path.join(script_dir, 'face_encodings', 'face_data.pkl')
    with open(face_data_path, 'wb') as f:
        pickle.dump(face_data, f)
    print(f"Saved face data: {len(face_data['names'])} faces")

def get_face_encodings(frame):
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) == 0:
        return [], [], []
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
    return face_encodings, face_locations, face_landmarks

def label_new_face(frame, face_encoding, face_location):
    new_face_data.put((frame, face_encoding, face_location))
    root.event_generate('<<NewFaceDetected>>')

def highlight_eyes(frame, face_landmarks):
    for landmarks in face_landmarks:
        for eye in ['left_eye', 'right_eye']:
            eye_points = landmarks[eye]
            center = np.mean(eye_points, axis=0).astype(int)
            size = int(np.max(np.linalg.norm(np.array(eye_points) - center, axis=1)))
            cv.circle(frame, tuple(center), size, (0, 255, 255), 2)

def process_face(frame, face_location, face_encoding, face_landmarks):
    top, right, bottom, left = face_location
    matches = face_recognition.compare_faces(face_data['encodings'], face_encoding)
    name = "Unknown"
    if True in matches:
        first_match_index = matches.index(True)
        name = face_data['names'][first_match_index]
    else:
        label_new_face(frame, face_encoding, face_location)
    
    highlight_eyes(frame, [face_landmarks])
    cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv.putText(frame, name, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam")
        return None
    return frame

def recognize_and_label_faces():
    global recognition_running, cap
    cap = cv.VideoCapture(0)
    while recognition_running:
        frame = capture_frame(cap)
        if frame is None:
            break
        
        face_encodings, face_locations, face_landmarks = get_face_encodings(frame)
        
        for face_encoding, face_location, face_landmark in zip(face_encodings, face_locations, face_landmarks):
            process_face(frame, face_location, face_encoding, face_landmark)
        
        if not frame_queue.full():
            frame_queue.put(frame)
    
    cap.release()
    cv.destroyAllWindows()

def update_frame():
    if recognition_running and not frame_queue.empty():
        frame = frame_queue.get()
        cv.imshow('Face Recognition', frame)
    if recognition_running:
        root.after(10, update_frame)

def create_data_view():
    global root, tree
    root = tk.Tk()
    root.title("Face Data Management")
    
    tree = ttk.Treeview(root, columns=('Name', 'Encoding'))
    tree.heading('Name', text='Name')
    tree.heading('Encoding', text='Encoding')
    tree.pack(expand=True, fill='both')
    
    update_data_view()
    
    view_button = ttk.Button(root, text="View Face", command=view_face)
    view_button.pack()
    
    delete_button = ttk.Button(root, text="Delete Face", command=delete_face)
    delete_button.pack()

    start_stop_button = ttk.Button(root, text="Start Viewer", command=toggle_viewer)
    start_stop_button.pack()

    quit_button = ttk.Button(root, text="Quit", command=quit_application)
    quit_button.pack()

    root.bind('<<NewFaceDetected>>', handle_new_face)

def update_data_view():
    for i in tree.get_children():
        tree.delete(i)
    for i, (name, encoding) in enumerate(zip(face_data['names'], face_data['encodings'])):
        tree.insert('', 'end', values=(name, f"Encoding {i+1}"))
    print(f"Updated data view: {len(face_data['names'])} faces")

def handle_new_face(event):
    frame, face_encoding, face_location = new_face_data.get()
    name = simpledialog.askstring("New Face", "Enter the name for the detected face:")
    if name:
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        image_path = os.path.join(script_dir, 'face_images', f"{name}.jpg")
        cv.imwrite(image_path, face_image)
        
        face_data['encodings'].append(face_encoding)
        face_data['names'].append(name)
        face_data['image_paths'].append(image_path)
        save_face_data()
        update_data_view()
        print(f"New face labeled: {name}")


def view_face():
    selected_items = tree.selection()
    if not selected_items:
        print("No face selected")
        return
    selected_item = selected_items[0]
    index = tree.index(selected_item)
    if index < len(face_data['image_paths']):
        image_path = face_data['image_paths'][index]
        if os.path.exists(image_path):
            face_image = cv.imread(image_path)
            face_image = cv.cvtColor(face_image, cv.COLOR_BGR2RGB)
            
            face_window = tk.Toplevel(root)
            face_window.title(face_data['names'][index])
            
            pil_image = Image.fromarray(face_image)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            label = ttk.Label(face_window, image=tk_image)
            label.image = tk_image
            label.pack()
        else:
            print(f"Image file not found: {image_path}")
    else:
        print(f"Image path not found for this face. Index: {index}, Images: {len(face_data['image_paths'])}")

def delete_face():
    selected_items = tree.selection()
    if not selected_items:
        print("No face selected for deletion")
        return
    selected_item = selected_items[0]
    index = tree.index(selected_item)
    
    if index < len(face_data['names']):
        name = face_data['names'][index]
        
        if 'image_paths' in face_data and index < len(face_data['image_paths']):
            image_path = face_data['image_paths'][index]
            if os.path.exists(image_path):
                os.remove(image_path)
            del face_data['image_paths'][index]
        
        del face_data['names'][index]
        del face_data['encodings'][index]
        
        save_face_data()
        update_data_view()
        print(f"Deleted face: {name}")
    else:
        print(f"Face data not found for deletion. Index: {index}")

def toggle_viewer():
    global recognition_running, viewer_thread, cap
    if not recognition_running:
        recognition_running = True
        viewer_thread = threading.Thread(target=recognize_and_label_faces)
        viewer_thread.start()
        root.after(0, update_frame)
        root.children['!button3'].config(text="Stop Viewer")
    else:
        recognition_running = False
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()
        if viewer_thread:
            viewer_thread.join()
        root.children['!button3'].config(text="Start Viewer")

def quit_application():
    global recognition_running
    recognition_running = False
    if viewer_thread:
        viewer_thread.join()
    root.quit()
    cv.destroyAllWindows()

def main():
    initialize_face_data()
    create_data_view()
    root.mainloop()

if __name__ == "__main__":
    main()