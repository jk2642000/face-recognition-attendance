import cv2
import numpy as np
import random
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from config import face_collection

# ========== GPU CONFIG ==========
print("[INFO] Checking GPU availability...")
print(f"TensorFlow version: {tf.__version__}")

try:
    gpus = tf.config.list_physical_devices('GPU')
    use_gpu = False
    
    if gpus and tf.test.is_built_with_cuda():
        try:
            # Configure GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 🔧 TUNE THIS: GPU memory limit in MB (adjust for your GPU)
            tf.config.experimental.set_memory_limit(gpus[0], 1536)  # 1.5GB for 2GB GPU
            use_gpu = True
            print(f"[GPU] Using GPU: {gpus[0].name}")
        except Exception as e:
            print(f"[WARNING] GPU setup failed: {e}")
            print("[INFO] Falling back to CPU")
            use_gpu = False
    else:
        if not tf.test.is_built_with_cuda():
            print("[WARNING] CUDA libraries not found")
        else:
            print("[WARNING] No GPU detected")
        print("[INFO] Using CPU")
        use_gpu = False
except Exception as e:
    print(f"[WARNING] GPU detection failed: {e}")
    print("[INFO] Using CPU")
    use_gpu = False

# ========== CONFIG - TUNABLE PARAMETERS ==========
# Number of face samples to capture per person
SAVE_COUNT = 10

# Enable face alignment for better accuracy
# Aligns faces based on eye positions before embedding
# Set to False if causing issues with your specific camera setup
USE_FACE_ALIGNMENT = True  # Set to False to disable alignment

def generate_unique_employee_id():
    while True:
        emp_id = ''.join([str(random.randint(0, 9)) for _ in range(12)])
        if not face_collection.find_one({"employee_id": emp_id}):
            return emp_id

def align_face(image, left_eye, right_eye):
    """Align face based on eye positions so eyes are horizontal
    This improves embedding consistency when faces are tilted
    """
    try:
        # Calculate angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Get image dimensions
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix and apply rotation
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return aligned
    except Exception as e:
        print(f"[WARNING] Face alignment failed: {e}")
        return image  # Return original image if alignment fails

def compare_faces(face1, face2):
    try:
        norm1 = np.linalg.norm(face1)
        norm2 = np.linalg.norm(face2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(face1, face2) / (norm1 * norm2)
    except Exception as e:
        print(f"[WARNING] Face comparison failed: {e}")
        return 0

def check_face_duplicates(test_embedding, strict_threshold=0.7, close_threshold=0.6):
    best_match = None
    best_similarity = 0
    
    # Group embeddings by person (employee_id)
    people_embeddings = {}
    
    try:
        # First, collect all embeddings grouped by employee_id
        for doc in face_collection.find():
            try:
                emp_id = doc.get("employee_id")
                if not emp_id:
                    continue
                    
                existing_embedding = np.array(doc.get("embedding", []), dtype=np.float32)
                if existing_embedding.size == 0:
                    continue
                
                if emp_id not in people_embeddings:
                    people_embeddings[emp_id] = {"embeddings": [], "doc": doc}
                
                people_embeddings[emp_id]["embeddings"].append(existing_embedding)
            except Exception as e:
                continue
        
        # Now compare against all embeddings for each person
        for emp_id, data in people_embeddings.items():
            try:
                # Compare with all embeddings for this person
                similarities = [compare_faces(test_embedding, emb) for emb in data["embeddings"]]
                if not similarities:
                    continue
                    
                # Use the maximum similarity for this person
                max_similarity = max(similarities)
                
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_match = data["doc"]
                    
                # If any embedding is above strict threshold, it's a duplicate
                if max_similarity >= strict_threshold:
                    return "duplicate", data["doc"], max_similarity
            except Exception as e:
                continue
                    
        if best_similarity >= close_threshold:
            return "close_match", best_match, best_similarity
            
        return "new_person", None, 0
        
    except Exception as e:
        print(f"[ERROR] Duplicate check failed: {e}")
        return "new_person", None, 0

def display_person_details(person_data, similarity=None):
    print("\n" + "="*50)
    print("EXISTING PERSON FOUND:")
    print("="*50)
    print(f"Name      : {person_data.get('name', 'N/A')}")
    print(f"Employee ID: {person_data.get('employee_id', 'N/A')}")
    print(f"Phone     : {person_data.get('phone', 'N/A')}")
    print(f"Address   : {person_data.get('address', 'N/A')}")
    
    birth_date = person_data.get('birth_date', '')
    if birth_date:
        try:
            from datetime import datetime
            birth_dt = datetime.strptime(birth_date, "%Y-%m-%d")
            today = datetime.now()
            age = today.year - birth_dt.year
            if today.month < birth_dt.month or (today.month == birth_dt.month and today.day < birth_dt.day):
                age -= 1
            print(f"Age       : {age} years")
        except:
            print(f"Birth Date: {birth_date}")
    
    gender_int = person_data.get('gender', 1)
    gender_text = "Female" if gender_int == 0 else "Male" if gender_int == 1 else "Other" if gender_int == 2 else "Unknown"
    print(f"Gender    : {gender_text}")
    
    if similarity:
        print(f"Face Match: {similarity:.3f} ({similarity*100:.1f}%)")
    print("="*50)

def get_user_choice(message, options):
    while True:
        try:
            print(f"\n{message}")
            for i, option in enumerate(options, 1):
                print(f"{i}. {option}")
            
            choice = input("\nEnter your choice (number): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(options):
                return choice_num - 1
            else:
                print(f"Please enter a number between 1 and {len(options)}")
                
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n[INFO] Registration cancelled")
            exit(0)

# User Input
try:
    person_name = input("Enter name: ").strip()
    if not person_name:
        print("[ERROR] Name cannot be empty")
        exit(1)
        
    phone = input("Enter phone number: ").strip()
    address = input("Enter address: ").strip()
except KeyboardInterrupt:
    print("\n[INFO] Registration cancelled")
    exit(0)
except Exception as e:
    print(f"[ERROR] Input error: {e}")
    exit(1)

# Gender Input
while True:
    try:
        gender_input = input("Enter gender (male/female/other): ").strip().lower()
        if gender_input == 'female':
            gender = 0
            break
        elif gender_input == 'male':
            gender = 1
            break
        elif gender_input == 'other':
            gender = 2
            break
        else:
            print("Please enter 'male', 'female', or 'other'")
    except KeyboardInterrupt:
        print("\n[INFO] Registration cancelled")
        exit(0)
    except Exception as e:
        print(f"[ERROR] Input error: {e}")

# Birth Date Input
while True:
    try:
        birth_date = input("Enter birth date (YYYY-MM-DD): ").strip()
        from datetime import datetime
        birth_datetime = datetime.strptime(birth_date, "%Y-%m-%d")
        
        if birth_datetime > datetime.now():
            print("Birth date cannot be in the future")
            continue
            
        age = datetime.now().year - birth_datetime.year
        if age < 0 or age > 120:
            print("Please enter a valid birth date (age should be 0-120 years)")
            continue
            
        break
    except ValueError:
        print("Please enter date in YYYY-MM-DD format (e.g., 1990-05-15)")
    except KeyboardInterrupt:
        print("\n[INFO] Registration cancelled")
        exit(0)
    except Exception as e:
        print(f"[ERROR] Date input error: {e}")

employee_id = generate_unique_employee_id()
print(f"[INFO] Assigned Employee ID: {employee_id}")

# Notify about face alignment status
if USE_FACE_ALIGNMENT:
    print("[INFO] Face alignment is ENABLED - faces will be aligned before processing")
    print("       To disable, set USE_FACE_ALIGNMENT = False in the config section")
else:
    print("[INFO] Face alignment is DISABLED")
    print("       To enable, set USE_FACE_ALIGNMENT = True in the config section")

# Initialize Models
print("\n[INFO] Loading models...")
try:
    detector = MTCNN()
    facenet = FaceNet()
    print("[SUCCESS] Models loaded successfully")
except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")
    exit(1)

# Initialize Camera
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        exit(1)
except Exception as e:
    print(f"[ERROR] Camera initialization failed: {e}")
    exit(1)

print(f"[INFO] Starting face capture for: {person_name}")

# Duplicate Detection
print("\n[INFO] Checking for duplicate faces...")
print("[INFO] Please look at the camera for face detection...")

test_embedding = None
detection_attempts = 0
max_attempts = 30

while test_embedding is None and detection_attempts < max_attempts:
    ret, frame = cap.read()
    if not ret:
        detection_attempts += 1
        continue
        
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    
    cv2.putText(frame, "Look at camera for duplicate check...", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Duplicate Check", frame)
    
    if faces:
        for face_data in faces:
            try:
                x, y, width, height = face_data.get('box', [0, 0, 0, 0])
                x, y = max(0, x), max(0, y)
                
                face_region = rgb[y:y+height, x:x+width]
                if face_region.size == 0:
                    continue
                
                # FACE ALIGNMENT: Align face using eye positions if enabled
                if USE_FACE_ALIGNMENT:
                    try:
                        # Get facial keypoints from MTCNN
                        keypoints = face_data.get('keypoints', {})
                        left_eye = keypoints.get('left_eye')
                        right_eye = keypoints.get('right_eye')
                        
                        # Safe fallback: Only align if BOTH eyes are detected
                        if left_eye and right_eye:
                            # Adjust keypoints to face region coordinates
                            left_eye = (left_eye[0] - x, left_eye[1] - y)
                            right_eye = (right_eye[0] - x, right_eye[1] - y)
                            face_region = align_face(face_region, left_eye, right_eye)
                        else:
                            # Log fallback to non-aligned processing
                            if not left_eye and not right_eye:
                                print(f"[INFO] Both eyes not detected - using non-aligned face")
                            elif not left_eye:
                                print(f"[INFO] Left eye not detected - using non-aligned face")
                            else:
                                print(f"[INFO] Right eye not detected - using non-aligned face")
                    except Exception as e:
                        print(f"[WARNING] Face alignment skipped: {e}")
                    
                face_resized = cv2.resize(face_region, (160, 160))
                face_array = np.expand_dims(face_resized, axis=0)
                test_embedding = facenet.embeddings(face_array)[0]
                print("[SUCCESS] Face detected for duplicate check")
                break
            except Exception as e:
                continue
    
    detection_attempts += 1
    if cv2.waitKey(100) & 0xFF == 27:
        print("[INFO] Duplicate check cancelled")
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

if test_embedding is None:
    print("[WARNING] Could not detect face for duplicate check")
    print("[INFO] Proceeding with registration (no duplicate check)")
else:
    status, matched_person, similarity = check_face_duplicates(test_embedding)
    
    if status == "duplicate":
        print("\n🚨 DUPLICATE FACE DETECTED! 🚨")
        display_person_details(matched_person, similarity)
        
        options = [
            "Update existing person's information",
            "Cancel registration (go back)"
        ]
        choice = get_user_choice("This face is already registered. What would you like to do?", options)
        
        if choice == 0:
            print("\n[INFO] Update feature coming soon. Registration cancelled.")
        else:
            print("[INFO] Registration cancelled")
        cap.release()
        cv2.destroyAllWindows()
        exit(0)
            
    elif status == "close_match":
        print("\n⚠️ SIMILAR FACE FOUND! ⚠️")
        print(f"Found a person with {similarity*100:.1f}% face similarity")
        display_person_details(matched_person, similarity)
        
        options = [
            "Continue with new registration (different person)",
            "Update existing person's information", 
            "Cancel registration (go back)"
        ]
        choice = get_user_choice("A similar face was found. What would you like to do?", options)
        
        if choice == 0:
            print("[INFO] Continuing with new registration...")
        elif choice == 1:
            print("\n[INFO] Update feature coming soon. Registration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
        else:
            print("[INFO] Registration cancelled")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
    else:
        print("[SUCCESS] No duplicate faces found. Proceeding with registration...")

save_index = 0

# Main Capture Loop
try:
    # For batch processing, we'll collect multiple face samples before processing
    face_batch = []
    face_metadata = []
    batch_size = min(3, SAVE_COUNT)  # Process up to 3 faces at once (adjust as needed)
    
    while cap.isOpened() and save_index < SAVE_COUNT:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)

            if faces and len(faces) > 0:
                # Process the first detected face (assuming it's the person being registered)
                face_data = faces[0]  # Just use the first face
                try:
                    x, y, width, height = face_data.get('box', [0, 0, 0, 0])
                    x, y = max(0, x), max(0, y)
                    
                    face_region = rgb[y:y+height, x:x+width]
                    if face_region.size == 0:
                        continue
                    
                    # FACE ALIGNMENT: Align face using eye positions if enabled
                    if USE_FACE_ALIGNMENT:
                        try:
                            # Get facial keypoints from MTCNN
                            keypoints = face_data.get('keypoints', {})
                            left_eye = keypoints.get('left_eye')
                            right_eye = keypoints.get('right_eye')
                            
                            # Safe fallback: Only align if BOTH eyes are detected
                            if left_eye and right_eye:
                                # Adjust keypoints to face region coordinates
                                left_eye = (left_eye[0] - x, left_eye[1] - y)
                                right_eye = (right_eye[0] - x, right_eye[1] - y)
                                face_region = align_face(face_region, left_eye, right_eye)
                            else:
                                # Log fallback to non-aligned processing
                                if not left_eye and not right_eye:
                                    print(f"[INFO] Both eyes not detected - using non-aligned face")
                                elif not left_eye:
                                    print(f"[INFO] Left eye not detected - using non-aligned face")
                                else:
                                    print(f"[INFO] Right eye not detected - using non-aligned face")
                        except Exception as e:
                            print(f"[WARNING] Face alignment skipped: {e}")
                        
                    # PREPROCESSING: Resize to FaceNet input size (160x160)
                    face_resized = cv2.resize(face_region, (160, 160))
                    
                    # Add to batch for processing
                    face_batch.append(face_resized)
                    face_metadata.append({
                        "sample_index": save_index
                    })
                    
                    # Process batch when it reaches the target size or we have enough samples
                    if len(face_batch) >= batch_size or save_index + len(face_batch) >= SAVE_COUNT:
                        # Convert to numpy array for batch processing
                        face_array_batch = np.array(face_batch)
                        
                        # Get embeddings for all faces in batch at once
                        embeddings = facenet.embeddings(face_array_batch)
                        
                        # Save each embedding to database
                        for i, (embedding, metadata) in enumerate(zip(embeddings, face_metadata)):
                            try:
                                face_collection.insert_one({
                                    "employee_id": employee_id,
                                    "name": person_name,
                                    "phone": phone,
                                    "address": address,
                                    "gender": gender,
                                    "birth_date": birth_date,
                                    "sample_index": metadata["sample_index"],
                                    "embedding": embedding.tolist()
                                })
                                print(f"[SAVED] Sample {metadata['sample_index']}")
                                save_index += 1
                            except Exception as e:
                                print(f"[ERROR] Failed to save to database: {e}")
                        
                        # Clear batch after processing
                        face_batch = []
                        face_metadata = []
                        cv2.waitKey(800)  # Brief pause between captures
                    
                    # If we've collected enough samples, exit
                    if save_index >= SAVE_COUNT:
                        break
                        
                except Exception as e:
                    print(f"[WARNING] Error processing face: {e}")
                    continue

            try:
                cv2.putText(frame, f"{save_index}/{SAVE_COUNT} Samples Captured", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.imshow("Register Face", frame)
            except Exception as e:
                pass
                
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        except Exception as e:
            continue
            
except KeyboardInterrupt:
    print("\n[INFO] Registration interrupted by user")
except Exception as e:
    print(f"[ERROR] Main loop error: {e}")

try:
    cap.release()
    cv2.destroyAllWindows()
    if save_index >= SAVE_COUNT:
        print("[SUCCESS] Face registration complete.")
    else:
        print(f"[INFO] Registration stopped. Captured {save_index}/{SAVE_COUNT} samples.")
except Exception as e:
    print(f"[WARNING] Cleanup error: {e}")