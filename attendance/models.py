import json
import os
from django.core.exceptions import ValidationError
from django.conf import settings
from django.db import models
from django.utils import timezone
from datetime import time, timedelta
from django.contrib.auth.models import User, Group
from django.utils.timezone import localtime, now
from django.db.models import Max
from django.db.models.signals import post_migrate
from django.dispatch import receiver
from PIL import Image
import face_recognition
import numpy as np
import requests

# Model to store the work hours of the company
# This model is used to set the default work hours for the company
# Ensure a default instance is created after migrations
@receiver(post_migrate)
def create_default_work_hours(sender, **kwargs):
    if sender.name == "attendance":  # Replace with your Django app name
        if not WorkHours.objects.exists():
            WorkHours.objects.create()

# Model to store the work hours of the company
# This model is used to set the default work hours for the company
class WorkHours(models.Model):
    open_time = models.TimeField(default="08:00:00", verbose_name="Opening Time")  # Default opening time
    close_time = models.TimeField(default="17:00:00", verbose_name="Closing Time")  # Default closing time

    class Meta:
        verbose_name = "Work Hours"
        verbose_name_plural = "Work Hours"

    def can_clock_in(self):
        """Returns True if employees are allowed to clock in, False otherwise."""
        current_time = localtime(now()).time()  # Get current local time
        return self.open_time <= current_time <= self.close_time

    def __str__(self):
        return f"Work Hours: {self.open_time} - {self.close_time}"

# Model to store the employee details
# This model is used to store the employee details such as name, contact number, and profile image
EMPLOYMENT_STATUS_CHOICES = [
    ('full_time', 'Full-Time'),
    ('part_time', 'Part-Time'),
    ('contract', 'Contractual'),
    ('probation', 'Probationary'),
    ('intern', 'Intern'),
    ('resigned', 'Resigned'),
    ('terminated', 'Terminated'),
    ('retired', 'Retired'),
]

class Employee(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, verbose_name="User Account")
    employee_number = models.CharField(max_length=8, unique=True, verbose_name="Employee Number")
    first_name = models.CharField(max_length=50, verbose_name="First Name")
    middle_name = models.CharField(max_length=50, blank=True, null=True, verbose_name="Middle Name")
    last_name = models.CharField(max_length=50, verbose_name="Last Name")
    gender = models.CharField(max_length=10, choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other')], verbose_name="Gender")
    birth_date = models.DateField(verbose_name="Date of Birth", null=True, blank=True)  # Optional field for date of birth
    hire_date = models.DateField(default=timezone.now, verbose_name="Hire Date")  # Date when the employee was hired
    email = models.EmailField(unique=True, verbose_name="Email Address")
    contact_number = models.CharField(max_length=15, verbose_name="Contact Number")
    group = models.ForeignKey(Group, related_name='employees', on_delete=models.SET_NULL, null=True, blank=True, verbose_name="Department")
    employment_status = models.CharField(max_length=20, choices=EMPLOYMENT_STATUS_CHOICES, default='full_time', verbose_name="Employee Status")
    profile_image = models.ImageField(upload_to='profiles/', blank=True, null=True, max_length=512, default='profiles/default_avatar.webp', verbose_name="Profile Image")
    hourly_rate  = models.BooleanField(default=False, verbose_name="Hourly Rate Employee")
    
    class Meta:
        verbose_name = "Employee"
        verbose_name_plural = "Employees"

    def save(self, *args, **kwargs):
        """ Automatically update is_staff and sync permissions if the user belongs to the Admin group """

        # Generate Employee Number if not already set (e.g. 2025000)
        # Only generate employee number if it doesn't already exist
        # Ensure hire_date is set
        if not self.hire_date:
            self.hire_date = timezone.now()

        # Generate Employee Number if not set
        if not getattr(self, "employee_number", None):
            
            if not Employee.objects.filter(employee_number=self.employee_number).exists():
                year_joined = self.hire_date.year

                # Find the max employee_num for that year
                last_employee = Employee.objects.filter(
                    employee_number__startswith=str(year_joined)
                ).aggregate(Max('employee_number'))['employee_number__max']

                if last_employee:
                    last_seq = int(last_employee[4:])  # Extract the ####
                else:
                    last_seq = 0  # First employee of that year

                # Generate new number
                self.employee_number = f"{year_joined}{last_seq + 1:04d}"
            
            if Employee.objects.filter(employee_number=self.employee_number).exists():
                pass

        # Check employment status and update is_active flag
        if self.employment_status in ['resigned', 'terminated', 'retired']:
            self.user.is_active = False  # Disable access for resigned/terminated/retired employees
        else:
            self.user.is_active = True  # Keep active for other statuses

        if self.group:
            # If assigned to Admin, set is_staff and assign permissions
            if self.group.name.upper() == "ADMIN" or self.group.name.upper() == "HR ADMIN":
                self.user.is_staff = True
            else:
                self.user.is_staff = False

            if "TEACHING" in self.group.name.upper() and not "NON-TEACHING" in self.group.name.upper():
                self.hourly_rate = True
            else:
                self.hourly_rate = False
                
            # Sync User's Groups and Permissions
            self.user.groups.set([self.group])  # Ensure user is assigned to the correct group
            self.user.user_permissions.set(self.group.permissions.all())  # Apply group permissions

        else:
            # If no group is assigned, remove all permissions
            self.hourly_rate = False
            self.user.groups.clear()
            self.user.user_permissions.clear()

            if self.user.is_superuser:
                self.user.is_staff = True
            else:
                self.user.is_staff = False

        # Save the user model first (after setting is_staff and permissions)
        self.user.save()

        # Ensure the default avatar is not duplicated
        if self.profile_image and self.profile_image.name != 'profiles/default_avatar.webp':
            # If an employee already has this image, prevent duplicate storage
            if Employee.objects.exclude(pk=self.pk).filter(profile_image=self.profile_image).exists():
                return  # Prevents duplicate image saving

        # Now save the employee model
        super().save(*args, **kwargs)

        # Process and resize image after saving
        if self.profile_image and self.profile_image.name != 'profiles/default_avatar.webp':
            self.resize_image(self.profile_image.path)

    def resize_image(self, image_path, size=(256, 256)):
        """Resize and crop image to 1:1 aspect ratio (square)"""
        try:
            img = Image.open(image_path)
            img = img.convert("RGB")  # Ensure image is in RGB format
            
            # Get the center crop for a perfect square
            width, height = img.size
            min_dim = min(width, height)
            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2
            img = img.crop((left, top, right, bottom))  # Crop to square

            # Resize image to required size
            img = img.resize(size, Image.LANCZOS)

            # Save the image (overwrite original)
            img.save(image_path, "JPEG", quality=90)
        
        except Exception as e:
            print(f"Error resizing image: {e}")

    @property
    def is_hourly_employee(self):
        """Check if employee is paid hourly."""
        return self.hourly_rate

    @property
    def avatar_url(self):
        if self.profile_image:
            return self.profile_image.url
        else:
            return os.path.join(settings.MEDIA_URL, 'profiles/default_avatar.png')
    

    def full_name(self):
        # Check if middle_name is None or empty and avoid including it
        if self.middle_name is not None:
            return f"{self.first_name} {self.middle_name} {self.last_name}"
        else:
            return f"{self.first_name} {self.last_name}"

    def __str__(self):
        return self.full_name()
    
    def total_hours_worked(self):
        """
        Calculate total hours worked.
        - Hourly employees: sum all logs.
        - Fixed employees: sum only one attendance per day (e.g., the latest or first).
        """
        total_hours = 0
        shift_records = ShiftRecord.objects.filter(employee=self)

        if self.hourly_rate:
            # Sum all records for hourly employees
            total_hours = sum(record.total_hours for record in shift_records)
        else:
            # For fixed employees, avoid duplicate daily logs by keeping the latest per day
            unique_days = {}
            for record in shift_records:
                day = record.date
                # Keep the latest record per day
                if day not in unique_days or record.id > unique_days[day].id:
                    unique_days[day] = record
            total_hours = sum(rec.total_hours for rec in unique_days.values())

        return total_hours

    def average_hours_worked(self):
        """Calculate average hours worked per shift."""
        shift_records = ShiftRecord.objects.filter(employee=self)
        if shift_records.count() == 0:
            return 0
        total_hours = self.total_hours_worked()
        return total_hours / shift_records.count()
    
    def get_employee_avatar(self):
        """Fetch Employee profile image or return default avatar"""
        if hasattr(self, "employee") and self.employee.profile_image:
            return self.employee.profile_image.url
        return "/media/profiles/default_avatar.webp"  # Default image

    # Monkey-patch the User model
    User.add_to_class("get_avatar", get_employee_avatar)

# Model to store multiple face images for facial recognition
class FaceImage(models.Model):
    employee = models.ForeignKey(Employee, related_name='face_images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='known_faces/', max_length=512)
    uploaded_at = models.DateTimeField(default=timezone.now)
    face_encoding = models.TextField(blank=True, null=True)
    
    class Meta:
        verbose_name = "Face Image"
        verbose_name_plural = "Face Images"

    def save(self, *args, **kwargs):
        """Override save method to store face encoding"""
        if not self.face_encoding:
            # If encoding doesn't exist, generate it from the image
            image_path = self.image.path
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)

            if encoding:
                self.face_encoding = json.dumps(encoding[0].tolist())  # Convert array to list and store as JSON
            else:
                self.face_encoding = None

        super().save(*args, **kwargs)

    def get_encoding(self):
        """Returns the decoded face encoding"""
        if self.face_encoding:
            return np.array(json.loads(self.face_encoding))  # Decode the JSON into a numpy array
        return None

    # This method is used to delete the image file when the instance is deleted
    def delete(self, *args, **kwargs):
        """Delete the image file when the instance is deleted."""
        if self.image:
            image_path = os.path.join(settings.MEDIA_ROOT, str(self.image))
            if os.path.exists(image_path):
                os.remove(image_path)
        super().delete(*args, **kwargs)

    
# Model to track attendance with clock in and out
class ShiftRecord(models.Model):
    ATTENDANCE_STATUSES = [
        ('EARLY', 'Early'),
        ('PRESENT', 'Present'),
        ('LATE', 'Late'),
        ('ABSENT', 'Absent'),
        ('HOLIDAY', 'Holiday'),
    ]

    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.localdate)
    
    # Clock-in and clock-out timestamps
    clock_in = models.DateTimeField(blank=True, null=True)
    clock_out = models.DateTimeField(blank=True, null=True)

    # Attendance status
    status = models.CharField(max_length=10, choices=ATTENDANCE_STATUSES, default='ABSENT')
    is_half_day = models.BooleanField(default=False, help_text="If checked, system will record as half-day with actual hours worked.")

    class Meta:
        verbose_name = "Attendance"
        verbose_name_plural = "Attendances"

    def __str__(self):
        return f"{self.employee.full_name()} Attendance on {self.date}"
    
    #Use for checking national holidays
    #If the date is a holiday, the attendance will be automatically marked as ABSENT without the need for HR intervention.
    def is_holiday(self, date):
        """Check if the date is a holiday using an external API."""
        current_year = timezone.now().year
        url = f"https://date.nager.at/api/v3/PublicHolidays/{current_year}/PH"
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            holidays = response.json()
            holiday_dates = [holiday["date"] for holiday in holidays]
            return str(date) in holiday_dates
        except requests.RequestException as e:
            print(f"Error fetching holidays: {e}")
            return False
        
    #This will be use to check if it sunday
    def is_sunday(self, date):
        """Check if the given date is a Sunday."""
        return date.weekday() == 6  # 6 is Sunday in Python's weekday() method

    @property
    def total_hours(self):
        """Calculate total hours worked for the shift."""
        if self.clock_in and self.clock_out:
            total = (self.clock_out - self.clock_in).total_seconds() / 3600
            return round(total, 2)  # Round to 2 decimal places for accuracy
        return 0
    
    def clean(self):
        """Override the clean method to enforce validation rules."""
        today = timezone.localdate()

        # Check if the current employee has an approved leave on the current date/day
        has_approved_leave = LeaveRequest.objects.filter(
            employee=self.employee,
            status='APPROVED',
            start_date__lte=self.date,
            end_date__gte=self.date
        ).exists()

        # Check if employee is hourly and if they have multiple attendance entries
        if not self.employee.is_hourly_employee:
            # For non-hourly employees, ensure they only have one record per day
            # Exclude self from the query to allow updating the same record
            existing_records = ShiftRecord.objects.filter(employee=self.employee, date=self.date)
            
            # Exclude self from the query to allow updating the same record
            if self.pk:
                existing_records = existing_records.exclude(pk=self.pk)

            if existing_records.exists():
                raise ValidationError(f"{self.employee.full_name()} already has attendance recorded for {self.date}. Only one entry is allowed per day.")

        # Check if the current employee has an approved leave on the current date/day don't allow them to clock-in/out
        if has_approved_leave:
            raise ValidationError(f"{self.employee.full_name()} has an approved leave on {self.date}.")

        # Check if the date is a Sunday
        if self.is_sunday(self.date):
            raise ValidationError("Clock-in is not allowed on Sundays. Please ensure that the date is a valid workday.")

        # Call the parent's clean method to ensure no other validations are skipped
        super().clean()
    
    def save(self, *args, **kwargs):
        """Override save to update attendance status and auto clock-out within valid office hours."""
        # If validation passes (no errors), then save as usual
        self.full_clean()  # This will trigger the `clean` method before saving
        
        today = timezone.localdate()

        # Check if this is a old record needed to be updated
        if ShiftRecord.objects.filter(pk=self.pk).exists():
            # If the record already exists, update the clock_in and clock_out times
            existing_record = ShiftRecord.objects.get(pk=self.pk)
            existing_record.clock_in = self.clock_in
            existing_record.clock_out = self.clock_out
        
        # If the record is new continue with the rest of the logic
        else:
            # Check if the date is a holiday
            if self.is_holiday(self.date):
                self.status = 'HOLIDAY'
                self.clock_in = None
                self.clock_out = None
                super().save(*args, **kwargs)
                
            # Get work hours from DB or use fallback defaults
            work_hours = WorkHours.objects.first()
            if not work_hours:
                opening_time = timezone.datetime.combine(today, time(8, 0))
                lunch_start = timezone.datetime.combine(today, time(12, 0))
                lunch_end = timezone.datetime.combine(today, time(13, 0))
                closing_time = timezone.datetime.combine(today, time(18, 0))  # 6:00 PM
            else:
                opening_time = timezone.datetime.combine(today, work_hours.open_time)
                lunch_start = timezone.datetime.combine(today, time(12, 0))
                lunch_end = timezone.datetime.combine(today, time(13, 0))
                closing_time = timezone.datetime.combine(today, work_hours.close_time)

            # Ensure time is timezone-aware
            opening_time = timezone.make_aware(opening_time, timezone.get_current_timezone())
            lunch_start = timezone.make_aware(lunch_start, timezone.get_current_timezone())
            lunch_end = timezone.make_aware(lunch_end, timezone.get_current_timezone())
            closing_time = timezone.make_aware(closing_time, timezone.get_current_timezone())

            # Define early and grace period thresholds
            early_threshold = opening_time - timedelta(hours=2)  # Can clock in up to 2 hours early
            grace_period = opening_time + timedelta(minutes=5)  # 5-minute grace period for "Present"

            if self.clock_in:
                # Ensure clock_in is timezone-aware
                if timezone.is_naive(self.clock_in):
                    self.clock_in = timezone.make_aware(self.clock_in)

                    
                # 🛠️ NEW: Only apply auto clock-out if employee is NOT hourly
                if not self.employee.hourly_rate:
                    # If Half-day is checked count only 4 hours else 8 hours
                    if not self.is_half_day:
                        auto_clock_out = self.clock_in + timedelta(hours=7)

                    if self.is_half_day:
                        auto_clock_out = self.clock_in + timedelta(hours=4)

                    # Check if clock_in is within the allowed range
                    if early_threshold <= self.clock_in < opening_time:
                        self.status = 'EARLY'
                    elif opening_time <= self.clock_in <= grace_period:
                        self.status = 'PRESENT'
                    elif self.clock_in > grace_period:
                        self.status = 'LATE'
                    else:
                        self.status = 'ABSENT'

                    # Add lunch break if applicable
                    if not self.is_half_day and self.clock_in < lunch_start and auto_clock_out >= lunch_start:
                        auto_clock_out += timedelta(hours=1)

                    # Limit to closing time
                    self.clock_out = min(auto_clock_out, closing_time)
                
                # Allow manual clock_out for hourly employees
                if self.employee.hourly_rate:
                    # ✅ Hourly: HR input for status and allow manual clock_out if given
                    self.status = self.status or 'PRESENT'  # Default fallback if HR forgets
                    self.clock_out = self.clock_out or None  # Optional, in case auto is undesired


        super().save(*args, **kwargs)

    
    
    def employee_full_name(self):
        return self.employee.full_name()  # Calls the Employee model's full_name method
    employee_full_name.short_description = "Employee Full Name"

    def employee_profile_picture(self):
        return self.employee.avatar_url  # Calls the Employee model's avatar_url property
    employee_profile_picture.short_description = "Employee Profile Picture"

# Model to store events (e.g., holidays, meetings) with recurrence support
# This model is used to store events with recurrence rules
from multiselectfield import MultiSelectField   
class Event(models.Model):
    DAY_CHOICES = [
        ('0', 'Sunday'),
        ('1', 'Monday'),
        ('2', 'Tuesday'),
        ('3', 'Wednesday'),
        ('4', 'Thursday'),
        ('5', 'Friday'),
        ('6', 'Saturday'),
    ]

    title = models.CharField(
        max_length=200,
        help_text="Enter a short title for the event (e.g., 'Team Meeting')."
    )
    description = models.TextField(
        help_text="Provide a detailed description of the event, including any important notes or agenda."
    )
    url = models.URLField(
        blank=True,
        null=True,
        help_text="Optional: Include a link for more information or an online meeting URL (e.g., Zoom or Google Meet)."
    )
    start = models.DateTimeField(
        help_text="Select the starting date and time of the event. Format: YYYY-MM-DD HH:MM."
    )
    end = models.DateTimeField(
        help_text="Select the ending date and time of the event. Format: YYYY-MM-DD HH:MM."
    )
    days_of_week = MultiSelectField(
        choices=DAY_CHOICES,
        blank=True,
        null=True,
        help_text=(
            "Optional: Select the days of the week this event should repeat on. "
            "Use this only if the event is recurring (e.g., every Monday and Wednesday). "
            "Leave blank for one-time events."
        )
    )
    all_day = models.BooleanField(
        default=False,
        help_text="Check this if the event lasts all day without a specific start or end time."
    )

    def __str__(self):
        return self.title

# Model to store camera details for attendance tracking
# This model is used to store camera details for attendance tracking

# Ensure a default instance is created after migrations
@receiver(post_migrate)
def create_default_camera(sender, **kwargs):
    if sender.name == "attendance":  # Replace with your app name
        if not Camera.objects.exists():
            Camera.objects.create(
                name="Default Camera",
                camera_url="0",  # Fallback for OpenCV (e.g. local webcam)
                mode="CHECK_IN",
                is_active=True
            )
class Camera(models.Model):
    MODE_CHOICES = [
        ('CHECK_IN', 'Check-In'),
        ('CHECK_OUT', 'Check-Out'),
    ]

    name = models.CharField(max_length=100, unique=True)
    camera_url = models.CharField(max_length=255, unique=True)
    mode = models.CharField(max_length=50, choices=MODE_CHOICES, default='CHECK_IN')
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = "Camera"
        verbose_name_plural = "Cameras"

    def __str__(self):
        return self.name
    
# Leave Request Model
# This model is used to store leave requests made by employees
class LeaveRequest(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('APPROVED', 'Approved'),
        ('REJECTED', 'Rejected'),
        ('CANCELLED', 'Cancelled'),
    ]

    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='leave_requests')
    start_date = models.DateField(verbose_name="Start Date")
    end_date = models.DateField(verbose_name="End Date")
    reason = models.TextField(verbose_name="Reason for Leave")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING', verbose_name="Leave Status")
    attachment = models.FileField(upload_to='leave_attachments/', null=True, blank=True, max_length=512)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Request Created At")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Last Updated At")
    approved_by = models.ForeignKey(Employee, on_delete=models.SET_NULL, null=True, blank=True, related_name='approved_leaves', verbose_name="Approved By")

    class Meta:
        verbose_name = "Leave Request"
        verbose_name_plural = "Leave Requests"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.employee.full_name()} - {self.start_date} to {self.end_date} ({self.status})"

    def duration(self):
        """Calculate the total number of days for the leave."""
        return (self.end_date - self.start_date).days + 1

    def is_currently_active(self):
        """Check if the leave is currently active."""
        today = timezone.localdate()
        return self.start_date <= today <= self.end_date and self.status == 'APPROVED'
    
    #If the application is Approved by the HR/ADMIN/Manager
    def save(self, *args, **kwargs):
        # Check if status is approved and approved_by is not set
        if self.status == 'APPROVED' and self.approved_by is None:
            # Look into the kwargs for user context
            request = kwargs.pop('request', None)
            if request and hasattr(request.user, 'employee'):
                self.approved_by = request.user.employee

        super().save(*args, **kwargs)
    
    
# Notification Model
# This model is used to store notifications for employees
class Notification(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, related_name='notifications')
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    def __str__(self):
        truncated = (self.message[:30] + '...') if len(self.message) > 30 else self.message
        timestamp = self.created_at.strftime('%b %d, %Y %I:%M %p')
        return f"{self.employee.full_name()} • {truncated} • {timestamp}"

# Model to store announcements for employees (HR/Admin)
# This model is used to store announcements for employees  
class Announcement(models.Model):
    title = models.CharField(max_length=255)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(Employee, on_delete=models.SET_NULL, null=True, related_name='announcements')
    is_active = models.BooleanField(default=True)  # Optional: for soft deleting/hiding

    class Meta:
        verbose_name = "Announcement"
        verbose_name_plural = "Announcements"

    def __str__(self):
        return self.title

# Model to store emergency contact details for employees
# This model is used to store emergency contact details for employees
class EmergencyContact(models.Model):
    employee = models.OneToOneField(Employee, on_delete=models.CASCADE, related_name='emergency_contact')
    contact_name = models.CharField(max_length=100, verbose_name="Contact Person")
    relationship = models.CharField(max_length=50, verbose_name="Relationship")
    phone_number = models.CharField(max_length=15, verbose_name="Phone Number")
    email = models.EmailField(blank=True, null=True, verbose_name="Email (optional)")
