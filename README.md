# Attendance Monitoring System

## Project Overview

This project implements a web-based attendance system using facial recognition technology tailored for St. Clare College of Caloocan employees. The system aims to replace the current manual logbook system, addressing issues of inaccuracy, inefficiency, and security concerns. It allows employees to log their time-in and time-out using facial recognition, enhancing the accuracy and security of attendance tracking.

## Scope

###   The system will:

* Automate employee attendance monitoring using facial recognition technology, primarily for employees with fixed salary rates.
* Capture and validate employees’ facial features to log their time-in and time-out records. 
* Store attendance logs in a database that can be accessed by authorized personnel. 
* Generate daily, weekly, and monthly attendance reports for monitoring and evaluation. 
* Support real-time verification to prevent proxy attendance or “buddy punching.” 
* Include an admin portal to manage employee records and attendance, with features to accommodate manual entries or adjustments for employees with flexible schedules. 
* Be deployed within the premises of St. Clare College of Caloocan and will function through IP cameras and a local network. 

###   The system will not:

* Support remote attendance for employees working off-site or from home.
* Integrate directly with payroll or HR systems, but will provide attendance reports that can be manually exported. 
* Accommodate student attendance monitoring. 
* Allow manual attendance entries by employees, except for authorized HR personnel.

## Functional Requirements

###   Attendance Automation:

* The system shall automate the time-in and time-out logging process using facial recognition primarily for employees with fixed salary rates.
* The system should provide flexibility for HR to record attendance for employees with flexible hours or per-hour schedules (e.g., faculty), potentially through manual entry or a separate interface within the system.
* The system shall aim to minimize manual attendance recording while accommodating the diverse scheduling needs of employees.

###   User Interface:

- The system shall provide a user-friendly interface for employees to check their attendance. 
- The system shall include an admin portal for managing employee records and settings. 

###   Data Management:

* The system shall store attendance logs in a secure database. 
* The system shall generate daily, weekly, and monthly attendance reports with advance filtering.

###   Multi-Factor Authentication (MFA):

* The system shall implement Multi-Factor Authentication (MFA) to enhance security.
* The system shall support WebAuthn for strong, passwordless authentication using hardware security keys or biometrics.
* The system shall support OAuth for secure authorization, allowing integration with other identity providers if needed.

###   Face Detection:

* The system shall employ different face detection mechanisms depending on the context:
     * **Frontend (Webcam Capture):** The system shall detect a face from the camera stream on the client-side (e.g., employee's workstation webcam) using the `face-api.js` library.
     * **Backend (IP Camera Feeds):** The system shall utilize the `face_recognition` library in Python to detect faces from IP camera feeds.
* If no face is detected in either scenario, the system shall continue attempting to detect a face until one is found.
* **For webcam capture in the UI,** the system may wait for a short interval (e.g., 2000 milliseconds) after a face is detected before sending the image data to the server for processing. This delay is a client-side optimization to ensure a stable image is captured via the browser's webcam API.

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

##   Non-Functional Requirements

###   Accuracy:

* The system must ensure accurate facial recognition to prevent attendance fraud.

###   Efficiency:

* The system must streamline the attendance process and reduce administrative workload. 
###   Security:

* The system must provide secure storage of attendance data and prevent unauthorized access. 

###   Reliability:

* The system must function consistently and reliably. 

###   Maintainability:

* The system should be easy to update and troubleshoot. 

##   Constraints
1.  **Network Dependency:** The system relies on a stable local network. 
2.  **Hardware Requirements:** Requires specific hardware like IP cameras and a server machine.
3.  **Environmental Factors:** Facial recognition accuracy can be affected by lighting and facial changes. 
4.  **Limited Integration:** Does not directly integrate with payroll or HR systems. 
5.  **On-Site Use:** Designed for on-site attendance monitoring within St. Clare College. 

##   Assumptions

1.  Employees are willing to use the facial recognition system.
2.  The system will be properly maintained to ensure continuous operation.
3.  Sufficient training will be provided to users.

##  Technologies Used

* **Face Recognition:** Utilizes facial recognition technology for automated attendance tracking. 
* **Database Management:** Employs a database to securely store attendance records. 
* **IP Cameras:** Uses IP cameras to capture employee facial data. 
* **Programming Languages/Frameworks:** Python, Django, OpenCV, face_recognition with DLIB

##  Getting Started

To get this project running, clone the repository and carefully follow the step-by-step installation guide in the [INSTALL.md](INSTALL.md) file.

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request with your changes or features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
