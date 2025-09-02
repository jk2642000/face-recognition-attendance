import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from config import face_collection, attendance_collection

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
            print("[INFO] Install CUDA Toolkit 11.2+ for GPU support")
        else:
            print("[WARNING] No GPU detected")
        print("[INFO] Using CPU")
        use_gpu = False
except Exception as e:
    print(f"[WARNING] GPU detection failed: {e}")
    print("[INFO] Using CPU")
    use_gpu = False

# ========== CONFIG - TUNABLE PARAMETERS ==========
# 🔧 TUNE THIS: Face similarity threshold (0.0-1.0)
# Higher = stricter matching, Lower = more lenient matching
THRESHOLD = 0.7 # Try: 0.5 (lenient) to 0.7 (balanced) - 0.8+ too strict!

# 🔧 TUNE THIS: Consecutive matches needed for recognition
# Higher = more stable but slower, Lower = faster but less stable
MATCH_REQUIRED = 5  # Try: 3 (fast) to 10 (stable)

# 🔧 TUNE THIS: Enable face alignment for better accuracy
# Aligns faces based on eye positions before embedding
# Set to False if causing issues with your specific camera setup
USE_FACE_ALIGNMENT = True  # Set to False to disable alignment

# 🔧 DEBUG MODE: Set to True to see similarity scores
DEBUG_MODE = True  # Set to False to hide debug output

# Display settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# ========== HELPERS ==========
def calculate_age(birth_date_str):
    """Calculate current age from birth date string - DYNAMIC AGE CALCULATION
    This automatically updates age based on current date vs birth date
    No need to update database - age changes automatically over time
    """
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        today = datetime.now()
        age = today.year - birth_date.year
        # Adjust if birthday hasn't occurred this year
        if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
            age -= 1
        return age
    except:
        return "Unknown"

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
    """Calculate similarity between two face embeddings using cosine similarity
    Returns: 0.0 (completely different) to 1.0 (identical faces)
    Higher values = more similar faces
    """
    try:
        # Cosine similarity: measures angle between two vectors
        # Perfect for comparing FaceNet embeddings
        norm1 = np.linalg.norm(face1)
        norm2 = np.linalg.norm(face2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(face1, face2) / (norm1 * norm2)
    except Exception as e:
        print(f"[WARNING] Face comparison failed: {e}")
        return 0

def mark_attendance(name):
    try:
        now = datetime.now()
        attendance_collection.insert_one({
            "name": name,
            "timestamp": now.isoformat()
        })
        print(f"[ATTENDANCE] Marked for {name}")
        
        if name in user_details:
            details = user_details[name]
            print(f"   - ID      : {details.get('employee_id', 'N/A')}")
            print(f"   - Age     : {details.get('age', 'N/A')}")
            print(f"   - Gender  : {details.get('gender', 'N/A')}")
            print(f"   - Phone   : {details.get('phone', 'N/A')}")
            print(f"   - Address : {details.get('address', 'N/A')}")
    except Exception as e:
        print(f"[ERROR] Failed to mark attendance for {name}: {e}")

# ========== INIT - AI MODELS ==========
# Loading face detection and recognition models
print("\n[INFO] Loading models...")
try:
    # MTCNN: Detects faces in images and extracts face regions
    detector = MTCNN()
    
    # FaceNet: Converts face images to 512-dimensional embeddings
    # These embeddings are mathematical representations of faces
    facenet = FaceNet()
    print("[SUCCESS] Models loaded successfully")
except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")
    exit(1)

# ========== LOAD DATABASE ==========
# Loading registered faces and user information from MongoDB
print("[INFO] Loading face data from MongoDB...")
people_faces = {}    # Dictionary: name -> list of face embeddings
user_details = {}    # Dictionary: name -> user info (age, gender, etc.)

try:
    for doc in face_collection.find():
        try:
            name = doc.get("name", "Unknown")
            if not name or name == "Unknown":
                print(f"[WARNING] Skipping document with missing name")
                continue
                
            embedding_data = doc.get("embedding")
            if not embedding_data:
                print(f"[WARNING] Skipping {name} - no embedding data")
                continue
                
            embedding = np.array(embedding_data, dtype=np.float32)
            if embedding.size == 0:
                print(f"[WARNING] Skipping {name} - invalid embedding")
                continue

            if name not in people_faces:
                people_faces[name] = []
                birth_date = doc.get("birth_date", "")
                current_age = calculate_age(birth_date) if birth_date else "Unknown"
                
                # Convert gender integer to text (Database: 0=Female, 1=Male, 2=Other)
                gender_int = doc.get("gender", 1)
                if gender_int == 0:
                    gender_text = "Female"
                elif gender_int == 1:
                    gender_text = "Male"
                elif gender_int == 2:
                    gender_text = "Other"
                else:
                    gender_text = "Unknown"
                
                user_details[name] = {
                    "phone": doc.get("phone", "N/A"),
                    "address": doc.get("address", "N/A"),
                    "employee_id": doc.get("employee_id", "UnknownID"),
                    "age": current_age,
                    "gender": gender_text
                }

            people_faces[name].append(embedding)
            
        except Exception as e:
            print(f"[WARNING] Error processing document: {e}")
            continue
            
except Exception as e:
    print(f"[ERROR] Database connection failed: {e}")
    print("[INFO] Starting with empty database")

print(f"[SUCCESS] Loaded {len(people_faces)} persons.")

# Notify about face alignment status
if USE_FACE_ALIGNMENT:
    print("[INFO] Face alignment is ENABLED - faces will be aligned before processing")
    print("       To disable, set USE_FACE_ALIGNMENT = False in the config section")
else:
    print("[INFO] Face alignment is DISABLED")
    print("       To enable, set USE_FACE_ALIGNMENT = True in the config section")

# ========== CAMERA SETUP ==========
# 🔧 TUNE THIS: Camera index (0=default, 1=external camera, etc.)
try:
    cap = cv2.VideoCapture(0)  # Try changing to 1, 2, etc. for different cameras
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        exit(1)
except Exception as e:
    print(f"[ERROR] Camera initialization failed: {e}")
    exit(1)
    
# Tracking variables
attendance_done = set()    # People who already marked attendance
match_count = {}           # Count consecutive matches per person

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            # Convert BGR (OpenCV) to RGB (MTCNN requirement)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # FACE DETECTION: Find all faces in current frame
            faces = detector.detect_faces(rgb)
            h, w, _ = frame.shape

            current_frame_matched = set()  # Track matches in this frame

            if faces:
                # BATCH PROCESSING: Prepare all faces for batch embedding
                face_regions = []
                face_metadata = []
                valid_face_indices = []
                
                # First pass: Extract and preprocess all faces
                for i, face_data in enumerate(faces):
                    try:
                        x, y, width, height = face_data.get('box', [0, 0, 0, 0])
                        x, y = max(0, x), max(0, y)
                        
                        # Extract face region from full image
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
                                elif DEBUG_MODE:
                                    # Log fallback to non-aligned processing
                                    if not left_eye and not right_eye:
                                        print(f"[INFO] Both eyes not detected - using non-aligned face")
                                    elif not left_eye:
                                        print(f"[INFO] Left eye not detected - using non-aligned face")
                                    else:
                                        print(f"[INFO] Right eye not detected - using non-aligned face")
                            except Exception as e:
                                if DEBUG_MODE:
                                    print(f"[WARNING] Face alignment skipped: {e}")
                        
                        # PREPROCESSING: Resize to FaceNet input size (160x160)
                        face_resized = cv2.resize(face_region, (160, 160))
                        
                        # Store face and metadata for batch processing
                        face_regions.append(face_resized)
                        face_metadata.append({
                            'x': x, 
                            'y': y, 
                            'width': width, 
                            'height': height,
                            'center_x': x + width//2,
                            'center_y': y + height//2
                        })
                        valid_face_indices.append(i)
                    except Exception as e:
                        print(f"[WARNING] Error preprocessing face: {e}")
                        continue
                
                # BATCH EMBEDDING: Process all faces at once if any valid faces found
                if face_regions:
                    try:
                        # Convert list to numpy array for batch processing
                        face_batch = np.array(face_regions)
                        
                        # Get all embeddings at once (much faster than one-by-one)
                        batch_embeddings = facenet.embeddings(face_batch)
                        
                        if DEBUG_MODE and len(face_regions) > 1:
                            print(f"[INFO] Batch processed {len(face_regions)} faces at once")
                    except Exception as e:
                        print(f"[ERROR] Batch embedding failed: {e}")
                        continue
                        
                    # Process each face with its embedding
                    for i, (curr_embedding, metadata) in enumerate(zip(batch_embeddings, face_metadata)):
                        try:
                            x = metadata['x']
                            y = metadata['y']
                            width = metadata['width']
                            height = metadata['height']
                            center_x = metadata['center_x']
                            center_y = metadata['center_y']
                            
                            label = "Unknown"
                            color = (0, 0, 255)
                            found = False
                            best_match_name = ""
                            best_similarity = 0

                            # FACE RECOGNITION: Compare against all registered faces
                            for name, embeddings in people_faces.items():
                                try:
                                    # Compare current face with all samples of this person
                                    similarities = [compare_faces(curr_embedding, emb) for emb in embeddings]
                                    if similarities:
                                        max_similarity = max(similarities)
                                        
                                        # Track best match for debugging
                                        if max_similarity > best_similarity:
                                            best_similarity = max_similarity
                                            best_match_name = name
                                        
                                        # 🔧 DEBUG: Show similarity scores
                                        if DEBUG_MODE:
                                            print(f"[DEBUG] {name}: {max_similarity:.3f} (threshold: {THRESHOLD})")

                                        # 🔧 RECOGNITION LOGIC: Similarity above threshold = potential match
                                        if max_similarity > THRESHOLD:
                                            match_count[name] = match_count.get(name, 0) + 1
                                            current_frame_matched.add(name)
                                            
                                            if DEBUG_MODE:
                                                print(f"[DEBUG] {name} match count: {match_count[name]}/{MATCH_REQUIRED}")

                                            # CONFIRMATION: Need consecutive matches for final recognition
                                            if match_count[name] >= MATCH_REQUIRED:
                                                details = user_details.get(name, {})
                                                emp_id = details.get("employee_id", "N/A")
                                                age = details.get("age", "N/A")  # Dynamic age!
                                                gender = details.get("gender", "N/A")
                                                
                                                # 🔧 TUNE THIS: Display format for recognized faces
                                                label = f"{name} ({age}y, {gender})"
                                                color = (0, 255, 0)  # Green for confirmed
                                                
                                                # ATTENDANCE: Mark once per session
                                                if name not in attendance_done:
                                                    mark_attendance(name)
                                                    attendance_done.add(name)
                                            else:
                                                # Still matching - show progress
                                                label = f"Matching... {match_count[name]}/{MATCH_REQUIRED}"
                                                color = (0, 255, 255)  # Yellow for matching

                                            found = True
                                            break
                                except Exception as e:
                                    print(f"[WARNING] Error matching {name}: {e}")
                                    continue

                            if not found and best_match_name and DEBUG_MODE:
                                print(f"[DEBUG] Best match: {best_match_name} ({best_similarity:.3f}) - Below threshold")
                                label = f"Close: {best_match_name} ({best_similarity:.2f})"
                                color = (128, 128, 255)

                            # VISUAL OUTPUT: Draw face detection box
                            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)

                            # VISUAL OUTPUT: Show recognition label above face
                            try:
                                text_size, _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
                                text_w, text_h = text_size
                                # Black background for text readability
                                cv2.rectangle(frame, (center_x - 10, center_y - 40), (center_x + text_w, center_y - 15), (0, 0, 0), -1)
                                # 🔧 TUNE THIS: Text color indicates status (red=unknown, yellow=matching, green=recognized)
                                cv2.putText(frame, label, (center_x, center_y - 20), FONT, FONT_SCALE, color, FONT_THICKNESS)
                            except Exception as e:
                                print(f"[WARNING] Error drawing label: {e}")
                        except Exception as e:
                            print(f"[WARNING] Error processing face: {e}")
                            continue

            # DECAY LOGIC: Reduce match count for people not seen in current frame
            # This prevents false positives from lingering matches
            for name in list(match_count.keys()):
                if name not in current_frame_matched:
                    old_count = match_count[name]
                    match_count[name] = max(0, match_count[name] - 1)
                    if DEBUG_MODE and old_count != match_count[name]:
                        print(f"[DEBUG] {name} count decayed: {old_count} -> {match_count[name]}")

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        except Exception as e:
            print(f"[WARNING] Frame processing error: {e}")
            continue
            
except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")
except Exception as e:
    print(f"[ERROR] Main loop error: {e}")

try:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released and windows closed")
except Exception as e:
    print(f"[WARNING] Cleanup error: {e}")