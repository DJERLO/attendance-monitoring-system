# Attendance Monitoring System

## Project Overview

The purpose of this project is to implement a web-based attendance tracking system for employees using facial recognition technology. The system allows employees to log their attendance remotely, either onsite or offsite, using a webcam or device camera and a web browser.

## Scope

### The system will:
- Detect faces using a real-time camera feed on the client side.
- Ensure anti-spoofing mechanisms are in place before recognizing the employee.
- Register attendance data (with timestamp) for recognized employees and log it to a database.
- Be accessible both onsite and remotely, allowing employees to log attendance from any location with an internet connection.

### The system will not:
- Handle attendance for non-registered users.
- Include multi-face recognition in a single frame.
- Store facial data locally on the client side.

## Functional Requirements

### Face Detection:
- The system shall detect a face from the camera stream on the client side using the `face-api.js` library.
- If no face is detected, the system shall continue streaming until a face is detected.
- Once a face is detected, the system shall wait for 2000 milliseconds before sending the image data to the server for processing.

### Anti-Spoofing:
- The system shall ensure that the detected face is real and not a spoof (such as a photo or video) using anti-spoofing algorithms trained on the [CASIA-FASD dataset](https://www.kaggle.com/datasets/minhnh2107/casiafasd).
- The image data will be passed to the anti-spoofing mechanism before proceeding to the recognition process.
- If the face is not real, the system shall reject the attempt and log an error.

### Facial Recognition:
- The system shall match the detected face with registered employees in the database using a facial recognition model implemented in Python.
- If a match is found, the system shall retrieve the corresponding employee’s information.
- If no match is found, the system shall reject the attendance attempt.

### Attendance Logging:
- The system shall log attendance for the recognized employee, recording the employee’s full name and the current timestamp in the attendance table.
- The system shall notify the employee that their attendance has been recorded.

### Remote Accessibility:
- The system shall allow employees to log attendance from any location with internet access, using their webcam.
- Attendance can be marked remotely or onsite as long as the employee's face is recognized by the system.

### Performance:
- The system must detect and recognize faces with a response time of less than 500 milliseconds.
- The system should only process and send face data to the server every 2000 milliseconds to prevent excessive server requests.

### Security:
- Facial data shall not be stored locally on the client side but processed in real-time for recognition.
- Data transferred between the client and server must be encrypted to ensure the security and privacy of employee information.

### Usability:
- The system interface must be user-friendly and accessible to employees without requiring technical knowledge.
- Employees shall receive feedback after logging attendance (e.g., confirmation messages).

### Availability:
- The system must be available 24/7, allowing employees working different shifts or remotely to log their attendance anytime.

## Constraints
- The system requires an internet connection for both client and server communication.
- Employees must be registered in the system before using the facial recognition attendance feature.
- The system will only send a new frame every 2000 milliseconds, ensuring no duplicate attendance entries for a single employee.

## Assumptions
- Employees have access to a web browser and a functioning camera to log their attendance.
- The system can be accessed remotely, but it assumes employees will have stable internet access.
- The facial recognition model is pre-trained and capable of accurately identifying registered employees from live camera feeds.

## Technologies Used
- **Face Detection**: Utilizes the `face-api.js` library for real-time face detection in the browser.
- **Anti-Spoofing**: Employs the CASIA-FASD dataset to train anti-spoofing algorithms, ensuring that detected faces are not images or videos.
- **Facial Recognition**: Implements a facial recognition model using Python to match detected faces with registered employee data.

## Getting Started

To get started with this project, clone the repository and follow the setup instructions in the [INSTALL.md](INSTALL.md) file.

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
