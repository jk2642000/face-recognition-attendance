# Face Recognition Attendance System

A robust face recognition system for registering users and marking attendance automatically using MTCNN face detection and FaceNet embeddings.

## Architecture Overview

### Core Technologies
- **Face Detection**: [MTCNN](https://github.com/ipazc/mtcnn) (Multi-task CNN for Joint Face Detection and Alignment)
- **Face Recognition**: [FaceNet](https://github.com/nyoki-mtl/keras-facenet) (Deep Learning Face Embedding)
- **Database**: MongoDB for storing face embeddings and user data
- **Computer Vision**: OpenCV for image processing
- **Deep Learning**: TensorFlow/Keras backend

### System Workflow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Feed   │───▶│   MTCNN Model    │───▶│  Face Detection │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Face Crops    │◀───│  Face Alignment  │◀───│   Eye Detection │
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │
          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ FaceNet Model   │───▶│ 512-D Embeddings│───▶│ Cosine Similarity│
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │                                              │
          ▼                                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MongoDB       │◀───│  Match Found?    │───▶│   Attendance    │
│   Storage       │    │  (Threshold>0.6) │    │    Marking      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Features

### Registration System
- Face-based user registration with duplicate detection
- Multiple face samples per person for improved accuracy
- Stores name, phone, address, gender, and birth date
- Automatic unique employee ID generation
- Prevents duplicate registrations with face similarity check

### Recognition System
- Real-time face detection and recognition
- Dynamic age calculation based on birth date
- Gender display (Female/Male/Other)
- Automatic attendance marking
- Visual feedback with color-coded labels
- Debug mode with similarity scores

### Technical Features
- GPU acceleration with CPU fallback
- MongoDB database integration
- Comprehensive error handling
- Tunable recognition parameters
- Multi-camera support

## Requirements

- Python 3.8+
- MongoDB 4.0+
- Webcam/Camera
- 4GB+ RAM (8GB recommended for GPU)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/jk2642000/face-recognition-attendance.git
cd face-recognition-attendance
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: If you encounter protobuf errors, run:
```bash
pip install protobuf==3.20.3
```

### 4. Install and Setup MongoDB

**Windows:**
1. Download MongoDB Community Server from [mongodb.com](https://www.mongodb.com/try/download/community)
2. Install and start MongoDB service
3. MongoDB will run on `mongodb://localhost:27017/`

**Linux:**
```bash
sudo apt update
sudo apt install mongodb
sudo systemctl start mongodb
```

**Mac:**
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb/brew/mongodb-community
```

### 5. Configure Database Connection
```bash
# Copy the template and edit with your settings
cp config_template.py config.py
```

Edit `config.py` with your MongoDB connection details:
```python
MONGODB_URI = 'mongodb://localhost:27017/'  # Update if needed
DATABASE_NAME = 'face_recognition'
```

## Usage

### Register a New Person
```bash
python register_face.py
```
- Enter personal details when prompted
- Look at the camera for duplicate check
- System will capture 10 face samples

### Recognize Faces and Mark Attendance
```bash
python recognize_face.py
```
- System will detect faces in real-time
- Recognized faces will be marked for attendance
- Press ESC to exit

## Technical Details

### Models Used

#### 1. MTCNN (Multi-task CNN)
- **Purpose**: Face detection and facial landmark detection
- **Output**: Face bounding boxes + 5 facial keypoints (eyes, nose, mouth corners)
- **Advantages**: High accuracy, handles multiple faces, provides facial landmarks
- **Paper**: [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)

#### 2. FaceNet
- **Purpose**: Face recognition through embedding generation
- **Architecture**: Inception-ResNet-v1 backbone
- **Output**: 512-dimensional face embeddings
- **Training**: Triplet loss for face verification
- **Paper**: [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

### Code Flow

#### Registration Process (`register_face.py`)
1. **User Input**: Collect personal information
2. **Duplicate Check**: 
   - Capture face using MTCNN
   - Generate embedding using FaceNet
   - Compare with existing faces in database
3. **Face Sampling**:
   - Capture 10 face samples for robustness
   - Apply face alignment using eye coordinates
   - Generate embeddings for each sample
4. **Database Storage**: Store embeddings + metadata in MongoDB

#### Recognition Process (`recognize_face.py`)
1. **Model Loading**: Initialize MTCNN + FaceNet models
2. **Database Loading**: Load all registered face embeddings
3. **Real-time Processing**:
   ```python
   # Face Detection
   faces = detector.detect_faces(rgb_frame)
   
   # Face Alignment (optional)
   aligned_face = align_face(face_region, left_eye, right_eye)
   
   # Embedding Generation
   embedding = facenet.embeddings(face_batch)
   
   # Similarity Matching
   similarity = cosine_similarity(new_embedding, stored_embedding)
   
   # Attendance Marking
   if similarity > THRESHOLD:
       mark_attendance(person_name)
   ```

### Key Algorithms

#### Face Alignment
```python
def align_face(face_img, left_eye, right_eye):
    # Calculate angle between eyes
    angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    # Rotate face to align eyes horizontally
    return rotate_image(face_img, angle)
```

#### Cosine Similarity
```python
def cosine_similarity(embedding1, embedding2):
    # Normalize embeddings
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    # Calculate cosine similarity
    return np.dot(embedding1, embedding2) / (norm1 * norm2)
```

## Configuration

You can adjust these parameters in the code:

- `THRESHOLD`: Face similarity threshold (0.6 recommended)
- `MATCH_REQUIRED`: Consecutive matches needed (5 recommended)
- `SAVE_COUNT`: Number of face samples to capture (10 default)
- `DEBUG_MODE`: Set to True to see similarity scores
- `USE_FACE_ALIGNMENT`: Enable/disable face alignment preprocessing

## GPU Support (Optional)

The system automatically detects and uses GPU if available. For GPU acceleration:

1. **Install CUDA Toolkit 11.2+**
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
   
2. **Install cuDNN**
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   
3. **Install TensorFlow GPU**
   ```bash
   pip install tensorflow[and-cuda]
   ```

## Troubleshooting

### Common Issues

**1. Protobuf Error:**
```bash
pip install protobuf==3.20.3
```

**2. Camera Not Found:**
- Check camera permissions
- Try different camera index (0, 1, 2) in code

**3. MongoDB Connection Error:**
- Ensure MongoDB service is running
- Check connection string in `config.py`

**4. Low Recognition Accuracy:**
- Ensure good lighting during registration
- Register multiple face angles
- Adjust `THRESHOLD` parameter (lower = stricter)

### Performance Tips

- Use GPU for faster processing
- Good lighting improves accuracy
- Register faces from multiple angles
- Keep camera stable during recognition

## Model Information

### Automatic Model Download
The models are automatically downloaded when you first run the code:

- **MTCNN**: Downloaded via `pip install mtcnn`
- **FaceNet**: Downloaded via `pip install keras-facenet`
  - Model: `facenet_keras.h5` (~92MB)
  - Architecture: Inception-ResNet-v1
  - Pre-trained on VGGFace2 dataset

### Model Performance
- **MTCNN**: ~95% face detection accuracy
- **FaceNet**: ~99.63% accuracy on LFW dataset
- **System**: ~90-95% recognition accuracy (depends on lighting/quality)

## Project Structure

```
face-recognition-attendance/
├── register_face.py          # User registration script
├── recognize_face.py         # Face recognition & attendance
├── config_template.py        # Database configuration template
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .gitignore               # Git ignore rules

# Auto-created during first run:
├── ~/.keras/models/          # FaceNet model cache
└── face_attendance/          # MongoDB database
    ├── face_db              # Face embeddings collection
    └── attendance_log       # Attendance records collection
```

## Database Schema

### Face Collection
```javascript
{
  "_id": ObjectId,
  "name": "John Doe",
  "employee_id": "EMP001",
  "phone": "+1234567890",
  "address": "123 Main St",
  "gender": 1,              // 0=Female, 1=Male, 2=Other
  "birth_date": "1990-01-01",
  "embedding": [0.1, 0.2, ...], // 512-dimensional array
  "timestamp": ISODate
}
```

### Attendance Collection
```javascript
{
  "_id": ObjectId,
  "name": "John Doe",
  "timestamp": "2025-01-01T09:00:00.000Z"
}
```

## Research Papers & References

1. **MTCNN**: [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)
2. **FaceNet**: [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
3. **Inception-ResNet**: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

## Performance Optimization

### For Better Accuracy
- Use good lighting during registration and recognition
- Capture faces from multiple angles during registration
- Ensure faces are clearly visible (not too far/close)
- Use higher resolution cameras
- Adjust `THRESHOLD` based on your security needs

### For Better Speed
- Enable GPU acceleration
- Use batch processing for multiple faces
- Optimize camera resolution (720p recommended)
- Enable face alignment only if needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/face-recognition-attendance.git

# Create development branch
git checkout -b feature/your-feature-name

# Install in development mode
pip install -e .
```

## Acknowledgments

- **MTCNN Implementation**: [ipazc/mtcnn](https://github.com/ipazc/mtcnn)
- **FaceNet Implementation**: [nyoki-mtl/keras-facenet](https://github.com/nyoki-mtl/keras-facenet)
- **Original FaceNet**: Google Research
- **MTCNN Paper**: Zhang et al., 2016

## License

MIT License - see LICENSE file for details

---

**Note**: This is an open-source project for educational and research purposes. For production use, consider additional security measures and compliance with local privacy laws.