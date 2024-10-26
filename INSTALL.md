# Attendance Monitoring System Installation Guide

This guide will help you set up the Attendance Monitoring System on your local machine. Please follow the instructions carefully to ensure a successful installation.

## Prerequisites

- Python 3.12 installed on your machine.
- A working internet connection.
- Basic knowledge of Django and web development.

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

## Troubleshooting

- Ensure all required packages are installed by checking your `requirements.txt`.
- If you encounter issues with database connections, double-check your `.env` configuration.
- Ensure your email server settings are correctly configured if you face email-related issues.

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
