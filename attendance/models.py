import os
from django.core.exceptions import ValidationError
from django.conf import settings
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

# Model to store the work hours of the company
# This model is used to set the default work hours for the company
class WorkHours(models.Model):
    open_time = models.TimeField(default="08:00:00")  # Default opening time
    close_time = models.TimeField(default="17:00:00")  # Default closing time

    def __str__(self):
        return f"Work Hours: {self.open_time} - {self.close_time}"

# Model to store the employee details
# This model is used to store the employee details such as name, contact number, and profile image
class Employee(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    employee_number = models.CharField(max_length=50, unique=True)
    first_name = models.CharField(max_length=50)
    middle_name = models.CharField(max_length=50, blank=True, null=True)
    last_name = models.CharField(max_length=50)
    email = models.EmailField(unique=True)
    contact_number = models.CharField(max_length=15)
    profile_image = models.ImageField(upload_to='profiles/', blank=True, null=True, default='profiles/default_avatar.webp')

    def save(self, *args, **kwargs):
        # Ensure the default avatar is not duplicated
        if self.profile_image and self.profile_image.name != 'profiles/default_avatar.webp':
            # Check if the image already exists
            if Employee.objects.filter(profile_image=self.profile_image).exists():
                raise ValidationError("This profile image already exists.")
        
        super(Employee, self).save(*args, **kwargs)

    @property
    def avatar_url(self):
        if self.profile_image:
            return self.profile_image.url
        else:
            return os.path.join(settings.MEDIA_URL, 'profiles/default_avatar.png')
    

    def full_name(self):
        return f"{self.first_name} {self.middle_name or ''} {self.last_name}"

    def __str__(self):
        return self.full_name()
    
    def total_hours_worked(self):
        """Calculate total hours worked across all shift records."""
        total_hours = 0
        shift_records = ShiftRecord.objects.filter(employee=self)

        for record in shift_records:
            total_hours += record.total_hours

        return total_hours

    def average_hours_worked(self):
        """Calculate average hours worked per shift."""
        shift_records = ShiftRecord.objects.filter(employee=self)
        if shift_records.count() == 0:
            return 0
        total_hours = self.total_hours_worked()
        return total_hours / shift_records.count()

# Model to store multiple face images for facial recognition
class FaceImage(models.Model):
    employee = models.ForeignKey(Employee, related_name='face_images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='known_faces/')
    uploaded_at = models.DateTimeField(default=timezone.now)

    
# Model to track attendance with clock in and out
class ShiftRecord(models.Model):
    ATTENDANCE_STATUSES = [
        ('PRESENT', 'Present'),
        ('LATE', 'Late'),
        ('ABSENT', 'Absent'),
    ]

    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    
    # Clock-in and clock-out timestamps
    clock_in = models.DateTimeField(blank=True, null=True)
    clock_out = models.DateTimeField(blank=True, null=True)

    # Attendance status
    status = models.CharField(max_length=10, choices=ATTENDANCE_STATUSES, default='ABSENT')

    def __str__(self):
        return f"{self.employee.full_name()} Attendance on {self.date}"

    @property
    def total_hours(self):
        """Calculate total hours worked for the shift."""
        if self.clock_in and self.clock_out:
            total = (self.clock_out - self.clock_in).total_seconds() / 3600
            return round(total, 2)  # Round to 2 decimal places for accuracy
        return 0
    
    
    def employee_full_name(self):
        return self.employee.full_name()  # Calls the Employee model's full_name method
    employee_full_name.short_description = "Employee Full Name"

    def employee_profile_picture(self):
        return self.employee.avatar_url  # Calls the Employee model's avatar_url property
    employee_profile_picture.short_description = "Employee Profile Picture"