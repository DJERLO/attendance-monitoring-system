# **Attendance Monitoring System - Installation Guide**  

This guide provides step-by-step instructions to set up the Attendance Monitoring System on your local machine. Please follow the instructions carefully.

---

## **Prerequisites**  

### **Software Requirements**
- **Python**: 3.12  
- **Django**: Installed via `pip install -r requirements.txt`  
- **Internet Connection**: Required for downloading dependencies  
- **Basic Knowledge**: Familiarity with Django and web development  

### **Hardware Used for Development**  
- **Camera**: A4TECH PK-635G Anti-glare USB Webcam with Built-in Microphone  
- **CPU**: [Intel® Core™ i3-9100F Processor](https://www.intel.com/content/www/us/en/products/sku/190886/intel-core-i39100f-processor-6m-cache-up-to-4-20-ghz/specifications.html)  
- **GPU**: GeForce GTX 1050 2G OC  

---

## **Model & Training Details**  

### **Model Selection**
- **ResNet-50**: A deep convolutional neural network (CNN) known for feature extraction and residual learning.  
- **Pre-trained Dataset**: **ImageNet1K** (1M+ images, 1000 categories)  
- **Fine-tuning Dataset**: **CASIA-FASD** (Real vs. spoofed faces) - [Download here](https://www.kaggle.com/datasets/minhnh2107/casiafasd)  

### **Purpose**
- The project enhances **facial recognition security** by distinguishing real human faces from spoofing attempts (e.g., printed photos, video replays, and 3D masks).  

📺 **[Watch Demo Video](https://github.com/user-attachments/assets/01a5fe4c-66f5-4362-b39b-65d6a23ee912)**  

---

## **Installation Steps**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/DJERLO/attendance-monitoring-system.git
cd attendance-monitoring-system
```

### **2. Create a Virtual Environment**  
```bash
python -m venv venv
```
#### **Activate the Virtual Environment**
- **Windows:**  
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux:**  
  ```bash
  source venv/bin/activate
  ```

### **3. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **4. Generate Django Secret Key**  
```python
import secrets
print(secrets.token_urlsafe(50))
```
Copy the generated key.

### **5. Configure Environment Variables**  

Create a `.env` file and add:  

```plaintext
# Django settings
SECRET_KEY='your-generated-secret-key'

# Google OAuth 2.0 credentials
GOOGLE_CLIENT_ID = ''
GOOGLE_CLIENT_SECRET = ''

FACEBOOK_CLIENT_ID = ''
FACEBOOK_CLIENT_SECRETE = ''

# Database Configuration (if using PostgreSQL)
DB_ENGINE=''
DB_NAME=''
DB_USER=''
DB_PASSWORD=''
DB_HOST=''
DB_PORT=''

#For Caching and Channels
REDIS_URL= ''

# Email Server
EMAIL_HOST_USER='your-email@gmail.com'
EMAIL_HOST_PASSWORD='your-email-password'
```

### **6. Apply Database Migrations**  
```bash
python manage.py migrate
```

### **7. Create a Superuser**  
```bash
python manage.py createsuperuser
```
Follow the prompts to create an admin account.

### **8. Run the Development Server**  
```bash
python manage.py runserver
```
Access the app at `http://127.0.0.1:8000/`.

### **9. Train Your Own Model (Optional)**  

📌 **Recommended for better performance**  

#### **Steps to Train ResNet-50**  

1. **Prepare Your Dataset**  
   - Download the [CASIA-FASD dataset](https://www.kaggle.com/datasets/minhnh2107/casiafasd).  
   - Ensure correct dataset structure.

2. **Update Dataset Path in `train_model.py`**  

3. **Start Training**  
   ```bash
   python attendance/train_model.py
   ```

4. **Model Output**  
   - Once training is complete, the model (`training_resnet50.pth`) is saved in the `model` directory.  

### **10. Access the Admin Interface**  
Go to `http://127.0.0.1:8000/admin/` and log in with the superuser credentials.

---

# **Building and Installing dlib with CUDA Support**  

This section guides you on compiling `dlib` with CUDA to accelerate face recognition.

## **Prerequisites**  

### **System Requirements**  
- **Windows 10/11** or **Linux**  
- **NVIDIA GPU** with CUDA Compute Capability **≥ 3.0**  
- **Minimum 4GB VRAM recommended**  

### **Software Requirements**  
| Dependency | Version |
|------------|---------|
| **Python** | 3.7+ |
| **CUDA Toolkit** | 11.x or 12.x |
| **cuDNN** | Compatible with CUDA |
| **CMake** | 3.18+ |
| **Visual Studio (Windows)** | 2019/2022 |
| **GCC (Linux)** | 7+ |
| **Boost Library (Optional)** | Latest |

### **Install Required Dependencies**  
```bash
pip install numpy cmake
```

---

## **Building dlib from Source with CUDA**  

### **1. Clone dlib Repository**  
```bash
git clone https://github.com/davisking/dlib.git
cd dlib
```

### **2. Configure CMake with CUDA**  
```bash
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
```
📌 **Windows users:** Use the correct generator:
```bash
cmake .. -G "Visual Studio 17 2022" -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
```

### **3. Compile dlib**  
- **Windows:**  
  ```bash
  cmake --build . --config Release
  ```
- **Linux/macOS:**  
  ```bash
  make -j$(nproc)
  ```

### **4. Install dlib**  
```bash
cd ..
python setup.py install
```

---

## **Verify CUDA Acceleration in dlib**  

Run the following test:  

```python
import dlib
print("CUDA Support:", dlib.DLIB_USE_CUDA)
print("Available GPU:", dlib.cuda.get_num_devices())
```
Expected output:  
```
CUDA Support: True
Available GPU: 1
```

---

## **Troubleshooting**  

### **1. CUDA Not Found Error**  
Ensure `CUDA_HOME` is correctly set:  
```bash
set CUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
```
Restart your terminal.

### **2. cuDNN Not Found**  
Make sure cuDNN is installed in:  
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64\cudnn.lib
```

### **3. CMake Cannot Find Visual Studio**  
Run:  
```bash
cmake --help
```
Ensure **Visual Studio 2019/2022** is available.

---

## **Final Notes**  
- **Ensure CUDA, cuDNN, and GPU drivers match versions.**  
- If you encounter issues, reinstall CUDA/cuDNN.  
- Consult [dlib's official documentation](http://dlib.net) for advanced configurations.  

---

# 🚀 Deploying a Django App Using ASGI (Daphne) on IIS with WebSockets Support

This tutorial will guide you through deploying a **Django app on IIS (Internet Information Services)** using **Daphne** and `httpPlatformHandler`. This setup is ideal for Django apps utilizing **ASGI**, especially when integrating **WebSockets**.

---

## ✅ Prerequisites

Before we begin, ensure we have the following:

- **Python 3.12+** installed with PATH configured
- Your Django app is ready (we'll clone an existing project for demonstration)
- **IIS** is installed and running on your system

---

## 📁 Example File Structure

We'll use my GitHub project: [`attendance-monitoring-system`](https://github.com/DJERLO/attendance-monitoring-system). Here's the simplified file structure of my project:

```
attendance-monitoring-system/
├── attendance/
│   ├── consumers.py         # WebSocket Consumers
│   ├── routing.py           # WebSocket routing
│   ├── templates/
│   └── views.py
├── attendance_system/
│   ├── asgi.py              # ASGI config
│   ├── settings.py
├── manage.py
├── requirements.txt
├── web.config               # Required by IIS
└── static/
```

---

## 🧠 Step-by-Step Deployment Guide

### 🔹 Step 1: Install Python 3.12 and Configure PATH

1. Download and install [Python 3.12](https://www.python.org/downloads/).
2. During installation, **check "Add Python to PATH"** to ensure Python is globally accessible in your system.

---

### 🔹 Step 2: Clone the Project and Install Requirements

Clone the project inside this directory **C:/inetpub/wwwroot** and install dependencies :

```bash
git clone https://github.com/DJERLO/attendance-monitoring-system
cd attendance-monitoring-system
pip install -r requirements.txt

# Optional if your database is SQLite or migrating to new one.
python manage.py makemigrations
python manage.py migrate
```

---

### 🔹 Step 3: Install IIS and Enable Handler Mappings

Open **PowerShell as Administrator** and run the following command to install IIS:

```powershell
Install-WindowsFeature -name Web-Server -IncludeManagementTools
```

Next, configure **Handler Mappings**:

1. Open **IIS Manager**.
2. Go to **Server Level > Feature Delegation**.
3. Click **Handler Mappings** and set it to **Read/Write**.

---

### 🔹 Step 4: Create a Website on IIS

1. Open **IIS Manager**.
2. Right-click **Sites > Add Website**.
3. Configure the following:
   - **Site Name**: `attendance_system`
   - **Physical Path**: `C:\inetpub\wwwroot\attendance-monitoring-system`
   - **Port**: `8001`


---

### 🔹 Step 5: Install HttpPlatformHandler

Download the **HttpPlatformHandler v1.2** from [IIS Downloads](https://www.iis.net/downloads/microsoft/httpplatformhandler).

Install it. This module allows IIS to proxy requests to platforms like Python or Node.js.

---

### 🔹 Step 6: Configure HttpPlatform for Django

In **IIS Manager**:

1. Go to **Sites > attendance_system > Configuration Editor**.
2. Navigate to `system.webServer/httpPlatform` and update the following settings:

| Key               | Value                                                                 |
|-------------------|-----------------------------------------------------------------------|
| `processPath`     | `C:\Users\Lorna\AppData\Local\Programs\Python\Python312\python.exe`    |
| `arguments`       | `C:\inetpub\wwwroot\attendance-monitoring-system\manage.py runserver %HTTP_PLATFORM_PORT%` |
| `stdoutLogEnabled`| `true`                                                               |
| `stdoutLogFile`   | `C:\inetpub\wwwroot\attendance-monitoring-system\logs`               |

Add an environment variable in the `<environmentVariables>` section:

```xml
<environmentVariable name="SERVER_PORT" value="%HTTP_PLATFORM_PORT%" />
```

---

### 🔹 Step 7: Configure App Settings

In **Configuration Editor**, set the following values under `system.webServer/appSettings`:

| Key                     | Value                                                             |
|-------------------------|-------------------------------------------------------------------|
| `PYTHONPATH`            | `C:\inetpub\wwwroot\attendance-monitoring-system`                 |
| `WSGI_HANDLER`          | `attendance_system.wsgi.application`                              |
| `DJANGO_SETTINGS_MODULE` | `attendance_system.settings`                                      |

---

### 🔹 Step 8: Add Module Mapping in IIS

Go to **IIS Manager** and follow these steps:

1. Navigate to **Sites > attendance_system > Handler Mappings**.
2. Click **Add Module Mapping** and enter the following values:

| Field                | Value                        |
|----------------------|------------------------------|
| Request Path         | `*`                          |
| Module               | `httpPlatformHandler`        |
| Name                 | `PythonHandlerHTTP`          |

🔧 Click **“Request Restrictions”** and **uncheck** the option:  
> “Invoke handler only if request is mapped to a file.”

---

### 🔹 Step 9: Create `web.config` in Root

In the root of your project (where `manage.py` is located), create a file named `web.config` with the following content:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <system.webServer>
        <httpPlatform 
            processPath="C:\Users\Lorna\AppData\Local\Programs\Python\Python312\python.exe" 
            arguments="C:\inetpub\wwwroot\attendance-monitoring-system\manage.py runserver %HTTP_PLATFORM_PORT%" 
            stdoutLogEnabled="true" 
            stdoutLogFile="C:\inetpub\wwwroot\attendance-monitoring-system\logs">
            
            <environmentVariables>
                <environmentVariable name="SERVER_PORT" value="%HTTP_PLATFORM_PORT%" />
            </environmentVariables>
        </httpPlatform>

        <handlers>
            <add name="PythonHandlerHTTP" path="*" verb="*" modules="httpPlatformHandler" resourceType="Unspecified" />
        </handlers>
    </system.webServer>

    <appSettings>
        <add key="PYTHONPATH" value="C:\inetpub\wwwroot\attendance-monitoring-system" />
        <add key="WSGI_HANDLER" value="attendance_system.wsgi.application" />
        <add key="DJANGO_SETTINGS_MODULE" value="attendance_system.settings" />
    </appSettings>
</configuration>
```

---

### 🧪 Final Step: Run and Test

Visit `http://localhost:8001` in your browser. Your Django app should be running! 🎉

---

## 💡 Notes for ASGI + Daphne (WebSockets)

For Django apps that use **WebSockets** and **ASGI**, you'll want to ensure you're using **Daphne** as the ASGI server.

You can configure this by replacing `runserver` with Daphne in your `web.config`:

```xml
arguments="daphne -b 127.0.0.1 -p %HTTP_PLATFORM_PORT% attendance_system.asgi:application"
```

Make sure **Daphne** is installed in your virtual environment:

```bash
pip install daphne
```
But in my case, In Django 4.2+ and above, you no longer need to explicitly run Daphne as a separate process during development. Instead, you can rely on `daphne` integrated into `INSTALLED_APPS`.

### Example in `settings.py`:

```python
INSTALLED_APPS = [
    "daphne",
    ...,
]

ASGI_APPLICATION = "attendance_system.asgi.application"
```

This will allow Django to automatically use Daphne for WebSocket support without additional arguments.

---

### 🔌 Testing WebSockets

Test your WebSocket connections locally or in production using **PieSocket WebSocket Tester**:

[PieSocket WebSocket Tester Extension](https://chromewebstore.google.com/detail/piesocket-websocket-teste/oilioclnckkoijghdniegedkbocfpnip)

---

## ✅ Done!

You’ve now configured and served a Django ASGI app using IIS and `httpPlatformHandler` with support for WebSockets via Daphne. You can customize it further by using Gunicorn, Channels, or reverse proxy with Nginx if needed.


---

## **Troubleshooting the Attendance System**  
- **Missing dependencies?** Run `pip install -r requirements.txt`.  
- **Database connection errors?** Check `.env` configurations.  
- **Email not working?** Verify SMTP settings.  

---

## **Contributing**  
Want to improve this project? Fork the repository and submit a pull request!

## **License**  
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file.

---