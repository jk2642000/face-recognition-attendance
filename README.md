# Face Recognition Attendance System

A robust face recognition system for registering users and marking attendance automatically using MTCNN face detection and FaceNet embeddings.

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
git clone https://github.com/yourusername/face-recognition-attendance.git
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

## Configuration

You can adjust these parameters in the code:

- `THRESHOLD`: Face similarity threshold (0.6 recommended)
- `MATCH_REQUIRED`: Consecutive matches needed (5 recommended)
- `SAVE_COUNT`: Number of face samples to capture (10 default)
- `DEBUG_MODE`: Set to True to see similarity scores

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

## Project Structure

```
face-recognition-attendance/
├── register_face.py          # User registration script
├── recognize_face.py         # Face recognition & attendance
├── config_template.py        # Database configuration template
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .gitignore               # Git ignore rules
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details