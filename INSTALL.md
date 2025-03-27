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
CLIENT_ID=''
CLIENT_SECRET=''

# Database Configuration (if using PostgreSQL)
DB_ENGINE=''
DB_NAME=''
DB_USER=''
DB_PASSWORD=''
DB_HOST=''
DB_PORT=''

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