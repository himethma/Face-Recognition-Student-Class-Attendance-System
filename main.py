from flask import Flask, jsonify, request
from pymongo import MongoClient

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb+srv://himethma:himethma123@iotcluster.vze4lf7.mongodb.net/?retryWrites=true&w=majority&appName=IoTcluster')
db = client["Face_Recognition"]
attendance_collection = db["Attendance data"]

@app.route("/attendance", methods=["GET"])
def get_attendance():
    """
    Retrieve all attendance records or filter by date.
    """
    date = request.args.get("date")  # Optional query parameter
    query = {"date": date} if date else {}
    records = list(attendance_collection.find(query, {"_id": 0}))
    return jsonify(records), 200

@app.route("/attendance", methods=["POST"])
def add_attendance():
    """
    Add a new attendance record.
    """
    data = request.json
    if not data.get("name") or not data.get("student_id"):
        return jsonify({"error": "Invalid data"}), 400
    
    attendance_collection.insert_one(data)
    return jsonify({"message": "Attendance record added"}), 201

@app.route("/attendance/<student_id>", methods=["GET"])
def get_student_attendance(student_id):
    """
    Get attendance for a specific student by ID.
    """
    records = list(attendance_collection.find({"student_id": student_id}, {"_id": 0}))
    return jsonify(records), 200

if __name__ == "__main__":
    app.run(debug=True)
