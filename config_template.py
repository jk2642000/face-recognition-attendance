# MongoDB Configuration Template
# Copy this file to 'config.py' and update with your MongoDB details

from pymongo import MongoClient

# MongoDB Connection Settings
# Replace with your MongoDB connection string
MONGODB_URI = 'mongodb://localhost:27017/'

# Database and Collection Names
DATABASE_NAME = 'face_recognition'
FACES_COLLECTION = 'faces'
ATTENDANCE_COLLECTION = 'attendance'

# Initialize MongoDB connection
try:
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    face_collection = db[FACES_COLLECTION]
    attendance_collection = db[ATTENDANCE_COLLECTION]
    
    # Test connection
    client.admin.command('ping')
    print("[SUCCESS] MongoDB connection established")
    
except Exception as e:
    print(f"[ERROR] MongoDB connection failed: {e}")
    print("Please check your MongoDB installation and connection string")
    exit(1)

# Optional: Create indexes for better performance
try:
    face_collection.create_index("name")
    face_collection.create_index("employee_id")
    attendance_collection.create_index("name")
    attendance_collection.create_index("timestamp")
except Exception as e:
    print(f"[WARNING] Index creation failed: {e}")