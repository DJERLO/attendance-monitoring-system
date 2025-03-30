import os
from django.core.exceptions import ValidationError
from django.conf import settings
from django.db import models
from django.utils import timezone
from datetime import time, timedelta
from django.contrib.auth.models import User, Group
from django.utils.timezone import localtime, now
from django.db.models.signals import post_migrate
from django.dispatch import receiver

# Model to store the work hours of the company
# This model is used to set the default work hours for the company
# Ensure a default instance is created after migrations
@receiver(post_migrate)
def create_default_work_hours(sender, **kwargs):
    if sender.name == "your_app_name":  # Replace with your Django app name
        if not WorkHours.objects.exists():
            WorkHours.objects.create()

class WorkHours(models.Model):
    open_time = models.TimeField(default="08:00:00", verbose_name="Opening Time")  # Default opening time
    close_time = models.TimeField(default="17:00:00", verbose_name="Closing Time")  # Default closing time

    def can_clock_in(self):
        """Returns True if employees are allowed to clock in, False otherwise."""
        current_time = localtime(now()).time()  # Get current local time
        return self.open_time <= current_time <= self.close_time

    def __str__(self):
        return f"Work Hours: {self.open_time} - {self.close_time}"

# Model to store the employee details
# This model is used to store the employee details such as name, contact number, and profile image

class Employee(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, verbose_name="User Account")
    employee_number = models.CharField(max_length=50, unique=True, verbose_name="Employee Number")
    first_name = models.CharField(max_length=50, verbose_name="First Name")
    middle_name = models.CharField(max_length=50, blank=True, null=True, verbose_name="Middle Name")
    last_name = models.CharField(max_length=50, verbose_name="Last Name")
    email = models.EmailField(unique=True, verbose_name="Email Address")
    contact_number = models.CharField(max_length=15, verbose_name="Contact Number")
    group = models.ForeignKey(Group, related_name='employees', on_delete=models.SET_NULL, null=True, blank=True, verbose_name="Department")
    profile_image = models.ImageField(upload_to='profiles/', blank=True, null=True, default='profiles/default_avatar.webp', verbose_name="Profile Image")

    class Meta:
        verbose_name = "Employee"
        verbose_name_plural = "Employees"

    def save(self, *args, **kwargs):
        """ Automatically update is_staff and sync permissions if the user belongs to the Admin group """
    
        if self.group:
            # If assigned to Admin, set is_staff and assign permissions
            if self.group.name.lower() == "admin":
                self.user.is_staff = True
            else:
                self.user.is_staff = False

            # Sync User's Groups and Permissions
            self.user.groups.set([self.group])  # Ensure user is assigned to the correct group
            self.user.user_permissions.set(self.group.permissions.all())  # Apply group permissions

        else:
            # If no group is assigned, remove all permissions
            self.user.is_staff = False
            self.user.groups.clear()
            self.user.user_permissions.clear()

        # Save the user model first (after setting is_staff and permissions)
        self.user.save()

        # Ensure the default avatar is not duplicated
        if self.profile_image and self.profile_image.name != 'profiles/default_avatar.webp':
            # If an employee already has this image, prevent duplicate storage
            if Employee.objects.exclude(pk=self.pk).filter(profile_image=self.profile_image).exists():
                return  # Prevents duplicate image saving

        # Now save the employee model
        super().save(*args, **kwargs)

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
    image = models.ImageField(upload_to='known_faces/')
    uploaded_at = models.DateTimeField(default=timezone.now)

    
# Model to track attendance with clock in and out
class ShiftRecord(models.Model):
    ATTENDANCE_STATUSES = [
        ('EARLY', 'Early'),
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
    
    def save(self, *args, **kwargs):
        """Override save to update attendance status and auto clock-out within valid office hours."""
        today = timezone.localdate()

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

            # Determine employee status based on clock-in time
            if early_threshold <= self.clock_in < opening_time:
                self.status = 'EARLY'
            elif opening_time <= self.clock_in <= grace_period:
                self.status = 'PRESENT'
            elif self.clock_in > grace_period:
                self.status = 'LATE'

            # Auto clock-out logic: exactly 8 hours after clock-in
            auto_clock_out = self.clock_in + timedelta(hours=7)

            # ✅ Fix: Add 1 hour only when shift crosses lunch
            if self.clock_in < lunch_start and auto_clock_out >= lunch_start:
                auto_clock_out += timedelta(hours=1)  

            # ✅ Fix: Only limit to closing time if it exceeds 6:00 PM
            # If clock-out is beyond closing time, stop at closing
            if auto_clock_out > closing_time:
                self.clock_out = closing_time
            else:
                self.clock_out = auto_clock_out

        super().save(*args, **kwargs)  # Save the instance

    
    
    def employee_full_name(self):
        return self.employee.full_name()  # Calls the Employee model's full_name method
    employee_full_name.short_description = "Employee Full Name"

    def employee_profile_picture(self):
        return self.employee.avatar_url  # Calls the Employee model's avatar_url property
    employee_profile_picture.short_description = "Employee Profile Picture"