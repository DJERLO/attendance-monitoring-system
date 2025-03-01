import os
from django.core.exceptions import ValidationError
from django.conf import settings
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

# Create your models here.
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
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    
    # Clock-in and clock-out timestamps
    clock_in_at_am = models.DateTimeField(blank=True, null=True)
    clock_out_at_am = models.DateTimeField(blank=True, null=True)
    clock_in_at_pm = models.DateTimeField(blank=True, null=True)
    clock_out_at_pm = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return f"{self.employee.full_name()} Attendance on {self.date}"

    @property
    def total_hours_at_morning(self):
        # Calculate total hours worked if all clock-ins and clock-outs are present
        total = 0
        if self.clock_in_at_am and self.clock_out_at_am:
            total += (self.clock_out_at_am - self.clock_in_at_am).seconds / 3600  # convert to hours
        return total
    
    @property
    def total_hours_at_afternoon(self):
        # Calculate total hours worked if all clock-ins and clock-outs are present
        total = 0
        if self.clock_in_at_pm and self.clock_out_at_pm:
            total += (self.clock_out_at_pm - self.clock_in_at_pm).seconds / 3600  # convert to hours
        return total
    
    @property
    def total_hours(self):
        # Calculate total hours worked if all clock-ins and clock-outs are present
        total = 0
        if self.clock_in_at_am and self.clock_out_at_am:
            total += (self.clock_out_at_am - self.clock_in_at_am).seconds / 3600  # convert to hours
        if self.clock_in_at_pm and self.clock_out_at_pm:
            total += (self.clock_out_at_pm - self.clock_in_at_pm).seconds / 3600  # convert to hours
        return total
    
    def employee_full_name(self):
        return self.employee.full_name()  # Calls the Employee model's full_name method
    employee_full_name.short_description = "Employee Full Name"

    def employee_profile_picture(self):
        return self.employee.avatar_url  # Calls the Employee model's avatar_url property
    employee_profile_picture.short_description = "Employee Profile Picture"