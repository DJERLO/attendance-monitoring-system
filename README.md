# Attendance Monitoring System

## Project Overview

The purpose of this project is to implement a web-based attendance tracking system for employees using facial recognition technology. The system allows employees to log their attendance remotely, either onsite or offsite, using a webcam or device camera and a web browser. User authentication is managed using Django Allauth, ensuring that only registered users can access the attendance functionality.

## Scope

### The system will:
- Detect faces using a real-time camera feed on the client side.
- Ensure anti-spoofing mechanisms are in place before recognizing the employee.
- Automatically record attendance data (with timestamp) for recognized employees without requiring explicit registration steps after facial recognition and log it to a database.
- Be accessible both onsite and remotely, allowing employees to log attendance from any location with an internet connection.

### The system will not:
- Handle attendance for non-registered users.
- Include multi-face recognition in a single frame.
- Store facial data locally on the client side.

## Functional Requirements

### User Authentication:
- The system shall use Django Allauth to manage user registrations and logins.
- Only authenticated users will have access to the attendance features.

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
- If a match is found, the system shall retrieve the corresponding employee’s information and automatically record their attendance.
- If no match is found, the system shall reject the attendance attempt.

### Attendance Logging:
- The system shall log attendance for the recognized employee, recording the employee’s full name and the current timestamp in the attendance table.
- The system shall notify the employee that their attendance has been recorded.

### Remote Accessibility:
- The system shall allow employees to log attendance from any location with internet access, using their webcam.
- Attendance can be marked remotely or onsite as long as the employee's face is recognized by the system.

## Non-Functional Requirements

### Performance:
- The system must detect and recognize faces with a response time of less than 500 milliseconds.
- The system should only process and send face data to the server every 2000 milliseconds to prevent excessive server requests.

### Security:
- Facial data shall not be stored locally on the client side but processed in real-time for recognition.
- Data transferred between the client and server must be encrypted to ensure the security and privacy of employee information.
- User authentication data will be securely managed using Django Allauth.

### Usability:
- The system interface must be user-friendly and accessible to employees without requiring technical knowledge.
- Employees shall receive feedback after logging attendance (e.g., confirmation messages).

### Availability:
- The system must be available 24/7, allowing employees working different shifts or remotely to log their attendance anytime.

### Constraints

1. **Device Compatibility**: The system is designed to work optimally on devices with modern web browsers (Chrome, Firefox, Safari, etc.) and may not function correctly on outdated browsers or devices.
  
2. **Camera Quality**: The effectiveness of the facial recognition feature depends on the quality of the camera used. Low-resolution cameras may result in inaccurate identification.

3. **Lighting Conditions**: The system requires adequate lighting for effective face detection and recognition. Poor lighting conditions may lead to failures in recognition.

4. **User Privacy**: The system must comply with relevant data protection regulations (such as GDPR or CCPA) concerning employee privacy and data handling.

5. **Server Load**: The system’s performance may be affected during peak usage times if the server cannot handle the volume of concurrent connections.

6. **Training Data Updates**: Regular updates and retraining of the facial recognition model may be necessary to maintain accuracy as employee appearances change over time.

7. **Network Latency**: A stable internet connection is crucial; high latency may impact the responsiveness of the attendance system.

8. **Hardware Limitations**: The performance of the facial recognition feature may vary based on the hardware specifications of the user's device.

### Assumptions

1. **User Familiarity**: It is assumed that employees are comfortable using web applications and have basic technical skills.

2. **Browser Permissions**: Users will grant the necessary permissions for camera access in their browsers.

3. **Registration Process**: Employees will complete the registration process correctly, including uploading a clear image for facial recognition.

4. **Camera Positioning**: It is assumed that users will set up their cameras in a position that allows for clear visibility of their faces.

5. **Stable Environment**: Employees will log their attendance from a stable environment without significant background movement that could interfere with facial recognition.

6. **Internet Bandwidth**: It is assumed that employees have sufficient internet bandwidth to support real-time video streaming for face detection.

7. **Software Updates**: Users will keep their web browsers and other relevant software updated to the latest versions for optimal functionality.

8. **Fallback Mechanism**: It is assumed there will be a fallback mechanism (such as manual attendance) in case the facial recognition system fails to recognize an employee.

## Technologies Used
- **User Authentication**: Utilizes Django Allauth for managing user accounts and authentication.
- **Face Detection**: Utilizes the `face-api.js` library for real-time face detection in the browser.
- **Anti-Spoofing**: Employs the CASIA-FASD dataset to train anti-spoofing algorithms, ensuring that detected faces are not images or videos.
- **Facial Recognition**: Implements a facial recognition model using Python to match detected faces with registered employee data.

## Getting Started

To get started with this project, clone the repository and follow the setup instructions in the [INSTALL.md](INSTALL.md) file.

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
