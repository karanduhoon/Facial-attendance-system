import os
import json
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from django.shortcuts import render
from .models import RegisteredUser
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import datetime
from django.http import JsonResponse
from django.core.files.base import ContentFile
import uuid  # To generate unique file names
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO

# Load the models
# model = InceptionResnetV1(pretrained='/Users/vismayparekh/Downloads/MLfinaproject 3/MLfinaproject/facial_attendance/attendance_app/best.pt').eval()
mtcnn = MTCNN(keep_all=False, device='cpu')
# model = torch.load('/Users/vismayparekh/Downloads/MLfinaproject 3/MLfinaproject/facial_attendance/attendance_app/best.pt')  # Replace with the actual path
# model.eval()


yolo_model = YOLO("/Users/vismayparekh/Downloads/MLfinaproject 3/MLfinaproject/facial_attendance/attendance_app/models/best.pt")

# Ensure the media directory exists
os.makedirs('media', exist_ok=True)

# Attendance log file
ATTENDANCE_LOG = os.path.join('media', 'attendance_logs.csv')

# Initialize log file
if not os.path.exists(ATTENDANCE_LOG):
    import pandas as pd
    pd.DataFrame(columns=["Name", "Timestamp", "Status"]).to_csv(ATTENDANCE_LOG, index=False)

def index(request):
    return render(request, 'index.html')


@csrf_exempt
def register_user(request):
    if request.method == 'POST':
        name = request.POST.get('name')

        # Check if an image is uploaded
        if 'image' in request.FILES:
            uploaded_file = request.FILES['image']
        else:
            return JsonResponse({'message': 'No image provided.'}, status=400)

        # Save image temporarily
        img_path = f"media/temp_{uuid.uuid4().hex}.jpg"
        with open(img_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Detect face using YOLO
        results = yolo_model.predict(source=img_path)
        if len(results[0].boxes) == 0:
            os.remove(img_path)  # Clean up temporary file
            return JsonResponse({'message': 'No face detected. Try again.'})

        # Get bounding box from YOLO results
        box = results[0].boxes.xyxy[0].cpu().numpy()  # Use the first detected face
        x1, y1, x2, y2 = map(int, box)

        # Crop and save the face
        image = cv2.imread(img_path)
        face = image[y1:y2, x1:x2]
        face_path = f"media/registered_faces/{uuid.uuid4().hex}.jpg"
        cv2.imwrite(face_path, face)

        # Register user
        user = RegisteredUser(
            name=name,
            image=uploaded_file,
            bounding_box=json.dumps(box.tolist())  # Save bounding box
        )
        user.save()

        os.remove(img_path)  # Clean up temporary file
        return JsonResponse({'message': f'User {name} registered successfully!'})
    return render(request, 'register.html')


# def mark_attendance(request):
#     if request.method == 'POST':
#         # Handle confirmation requests
#         if request.content_type == "application/json":
#             data = json.loads(request.body)
#             if data.get("confirm") and "name" in data:
#                 # Mark attendance for confirmed name
#                 timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                 attendance_log_path = "media/attendance_logs.csv"
#                 log_entry = pd.DataFrame([[data["name"], timestamp, "Present"]], columns=["Name", "Timestamp", "Status"])

#                 if os.path.exists(attendance_log_path):
#                     log = pd.read_csv(attendance_log_path)
#                     log = pd.concat([log, log_entry], ignore_index=True)
#                 else:
#                     log = log_entry

#                 log.to_csv(attendance_log_path, index=False)
#                 return JsonResponse({'message': f"Attendance marked for {data['name']}. Thank you!"})

#             return JsonResponse({'message': 'Attendance confirmation failed.'}, status=400)

#         # Check if an image is uploaded
#         if 'image' not in request.FILES:
#             return JsonResponse({'message': 'No photo uploaded or captured yet.'}, status=400)

#         # Save uploaded image temporarily
#         uploaded_file = request.FILES['image']
#         temp_image_path = f"media/temp_image.jpg"
#         with open(temp_image_path, 'wb') as f:
#             f.write(uploaded_file.read())

#         # Detect faces using MTCNN
#         image = cv2.imread(temp_image_path)
#         boxes, _ = mtcnn.detect(image)

#         if boxes is not None:
#             recognized_users = []
#             for box in boxes:
#                 x1, y1, x2, y2 = [int(coord) for coord in box]
#                 face = image[y1:y2, x1:x2]

#                 # Generate face embedding
#                 face = cv2.resize(face, (160, 160))
#                 face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0) / 255.0
#                 face = (face - 0.5) / 0.5
#                 face = face.type(torch.FloatTensor)
#                 with torch.no_grad():
#                     embedding = model(face).cpu().numpy()

#                 # Compare embedding with registered users
#                 users = RegisteredUser.objects.all()
#                 for user in users:
#                     saved_embedding = np.array(json.loads(user.face_embedding))
#                     similarity = cosine_similarity(embedding, saved_embedding)
#                     if similarity[0][0] > 0.8:  # Match threshold
#                         if user.name not in recognized_users:
#                             recognized_users.append(user.name)

#             os.remove(temp_image_path)

#             if recognized_users:
#                 # Mark attendance for all recognized users
#                 timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                 attendance_log_path = "media/attendance_logs.csv"
#                 log_entries = [[name, timestamp, "Present"] for name in recognized_users]

#                 if os.path.exists(attendance_log_path):
#                     log = pd.read_csv(attendance_log_path)
#                     new_logs = pd.DataFrame(log_entries, columns=["Name", "Timestamp", "Status"])
#                     log = pd.concat([log, new_logs], ignore_index=True)
#                 else:
#                     log = pd.DataFrame(log_entries, columns=["Name", "Timestamp", "Status"])

#                 log.to_csv(attendance_log_path, index=False)
#                 return JsonResponse({'message': 'Attendance marked for: ' + ', '.join(recognized_users)})

#             return JsonResponse({'message': 'No recognized faces in the photo. Please register first.'})

#         else:
#             os.remove(temp_image_path)
#             return JsonResponse({'message': 'No face detected in the photo.'})

#     return render(request, 'mark_attendance.html')


@csrf_exempt
def mark_attendance(request):
    if request.method == 'POST':
        # Check if an image is uploaded
        if 'image' not in request.FILES:
            return JsonResponse({'message': 'No image provided.'}, status=400)

        # Save uploaded image temporarily
        uploaded_file = request.FILES['image']
        temp_image_path = f"media/temp_image.jpg"
        with open(temp_image_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Detect face using YOLO
        results = yolo_model.predict(source=temp_image_path)
        if len(results[0].boxes) == 0:
            os.remove(temp_image_path)
            return JsonResponse({'message': 'No face detected. Try again.'})

        # Get bounding box and check similarity
        recognized_users = []
        image = cv2.imread(temp_image_path)
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            face = image[y1:y2, x1:x2]
            for user in RegisteredUser.objects.all():
                # Compare with stored bounding boxes or embeddings
                saved_box = np.array(json.loads(user.bounding_box))
                iou = calculate_iou(box, saved_box)
                if iou > 0.5:  # Example threshold for matching
                    recognized_users.append(user.name)

        os.remove(temp_image_path)
        if recognized_users:
            return JsonResponse({'message': f'Attendance marked for: {", ".join(recognized_users)}'})
        return JsonResponse({'message': 'No matching face detected.'})
    return render(request, 'mark_attendance.html')

def calculate_iou(box1, box2):
    """Calculate Intersection Over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union

