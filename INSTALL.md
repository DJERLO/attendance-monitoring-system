# Attendance Monitoring System Installation Guide

This guide will help you set up the Attendance Monitoring System on your local machine. Please follow the instructions carefully to ensure a successful installation.

## Prerequisites

- Python 3.12 installed on your machine.
- A working internet connection.
- Basic knowledge of Django and web development.

### Hardware Used for Development

#### **Hardware Setup:**
- **Camera**: A4TECH PK-635G Anti-glare USB Webcam with Built-in Microphone
- **CPU**: [Intel® Core™ i3-9100F Processor](https://www.intel.com/content/www/us/en/products/sku/190886/intel-core-i39100f-processor-6m-cache-up-to-4-20-ghz/specifications.html)
- **GPU**: GeForce GTX 1050 2G OC

### Model & Training Details

- **Model**: For our anti-spoofing solution, we utilized **ResNet-50**, a powerful deep convolutional neural network (CNN) known for its ability to learn complex features through residual learning.

- **Pre-trained Dataset**: The model was initially pre-trained on **ImageNet1K**, which includes over 1 million images across 1,000 categories. This pre-training helps the model capture a wide variety of features and enhances performance on subsequent tasks.

- **Fine-tuning Dataset**: We fine-tuned the ResNet-50 model using the **CASIA-FASD** dataset, which features close-up images of both real faces and various spoofing attacks. This dataset is crucial for our project, as it includes examples of printed photos, video replays, and 3D masks. You can access the dataset [here on Kaggle](https://www.kaggle.com/datasets/minhnh2107/casiafasd).

- **Project Purpose**: The primary objective of this project is to improve security in face recognition systems by accurately distinguishing between genuine human faces and spoofing attempts. The ResNet-50 architecture's capacity to analyze subtle differences in features makes it especially well-suited for this application.

For a demonstration of how anti-spoofing works in recognizing spoofing attacks within the attendance system using facial recognition technologies, please refer to the video below:

**Reference Video**:  
[Anti-Spoofing Example](https://github.com/user-attachments/assets/01a5fe4c-66f5-4362-b39b-65d6a23ee912)

## Steps to Set Up

### 1. Clone the Repository

Open your terminal and clone the repository using the following command:

```bash
git clone https://github.com/DJERLO/attendance-monitoring-system.git
cd attendance-monitoring-system
```

### 2. Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies. You can create one using `venv`:

```bash
python -m venv venv
```

Activate the virtual environment:

- **On Windows:**
  ```bash
  venv\Scripts\activate
  ```

- **On macOS and Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

### 4. Generate Django Secret Key

To enhance the security of your Django application, you will need to generate a secret key. You can use the following method:

1. Open a Python shell:
   ```bash
   python
   ```

2. Run the following code to generate a secret key:
   ```python
   import secrets
   print(secrets.token_urlsafe(50))
   ```

3. Copy the generated key.

### 5. Create a `.env` File

Create a file named `.env` in the root of your project and add the following configuration, including your generated secret key:

```plaintext
# Django settings
SECRET_KEY='your-generated-secret-key'  # Set your Django secret key here

# Google OAuth 2.0 credentials for application authentication
CLIENT_ID=''  # Your Google OAuth client ID
CLIENT_SECRET=''  # Your Google OAuth client secret

# Firebase Admin SDK configuration
FIREBASE_CREDENTIALS=''  # Path to your Firebase Admin SDK JSON credentials file
FIREBASE_STORAGE_BUCKET=''  # Your Firebase storage bucket URL
FIREBASE_DATABASE_URL=''  # Your Firebase database URL

# Database (if using PostgreSQL or another DB)
DB_ENGINE=''  # Database engine, e.g., django.db.backends.postgresql
DB_NAME=''  # Name of your database
DB_USER=''  # Database user
DB_PASSWORD=''  # Database password
DB_HOST=''  # Database host, e.g., localhost
DB_PORT=''  # Database port, e.g., 5432 for PostgreSQL

# Email Server
EMAIL_HOST_USER='your-email@gmail.com'
EMAIL_HOST_PASSWORD='your-email-password'
```

Make sure to fill in the values with your own credentials.

### 6. Configure Database Settings

If you are using PostgreSQL or another database, ensure that you have created the database specified in your `.env` file. Update the database connection settings accordingly.

### 7. Run Database Migrations

After configuring your database, apply the migrations to set up the necessary database tables:

```bash
python manage.py migrate
```

### 8. Create a Superuser

You can create a superuser to access the Django admin interface:

```bash
python manage.py createsuperuser
```

Follow the prompts to create a superuser account.

### 9. Run the Development Server

Start the Django development server using:

```bash
python manage.py runserver
```

You can now access the application at `http://127.0.0.1:8000/` in your web browser.

### 10. Access the Admin Interface

To access the admin interface, go to `http://127.0.0.1:8000/admin/` and log in with the superuser credentials you created earlier.

### 11. Train Your Own Model (Optional, but Recommended)

While the model files are available in the repository, we recommend that users train their own model to ensure optimal performance based on their specific hardware configurations. Follow these steps to generate your own `training_resnet50.pth` file:

1. **Prepare Your Dataset**
   - We encourage you to explore Kaggle for suitable datasets, such as the CASIA-FASD dataset, which contains images labeled as 'real' and 'fake'. You can download it directly from Kaggle using the following command:
   ```python
   path = kagglehub.dataset_download("minhnh2107/casiafasd")
   ```
   - Ensure you have a dataset structured correctly for the model training script.

2. **Update the Dataset Path**
   - Open the training script located at `attendance/train_model.py` and ensure the path to your dataset is correctly set in the script configuration. Since datasets may have different structures, adjust the paths according to your dataset.

3. **Execute the Training Script**
   - Run the following command in your terminal to start the training process:
   ```bash
   python attendance/train_model.py
   ```

4. **Monitor Training Progress**
   - The script will display training loss and other metrics in the terminal. Keep an eye on these outputs to track the training process.

5. **Completion**
   - Once training is complete, your trained model will be saved in the `model` directory. You can now use this model for predictions in the Attendance Monitoring System.

### Important Note
- Although the pre-trained model is available for use, training your own version can lead to better accuracy tailored to your hardware specifications. If you choose to change the dataset or use your own, please ensure you understand the dataset structure and how it integrates with the training script to avoid errors.

### Additional Notes
- The training process may take time depending on your hardware specifications. Ensure your environment is set up for optimal performance.
- If you have any questions or encounter issues while training the model, feel free to reach out for help or consult the documentation for more details.

## Troubleshooting

- Ensure all required packages are installed by checking your `requirements.txt`.
- If you encounter issues with database connections, double-check your `.env` configuration.
- Ensure your email server settings are correctly configured if you face email-related issues.

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
