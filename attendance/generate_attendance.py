import os
import sys
import django
from django.utils import timezone

# Get the project root directory (one level up from this script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to Python path
sys.path.append(BASE_DIR)

# Set Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system.settings")  # Change to your actual project name

# Initialize Django
django.setup()

# Import models AFTER setting up Django
from attendance.models import Employee, ShiftRecord

def generate_attendance():
    today = timezone.now().date()

    # Skip Sundays
    if today.weekday() == 6:
        return  # Exit early

    for employee in Employee.objects.all():
        if not ShiftRecord.objects.filter(employee=employee, date=today).exists():
            ShiftRecord.objects.create(employee=employee, date=today, status='ABSENT')

    print(f"Attendance records generated for {today}")

if __name__ == "__main__":
    generate_attendance()
