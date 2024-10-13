import os
from django.core.exceptions import ValidationError
from django.conf import settings
from django.db import models
from django.utils import timezone

# Create your models here.
class Employee(models.Model):
    employee_id = models.CharField(max_length=50, unique=True)
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
    
# Model to store multiple face images for facial recognition
class FaceImage(models.Model):
    employee = models.ForeignKey(Employee, related_name='face_images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='known_faces/')
    uploaded_at = models.DateTimeField(default=timezone.now)

    
class Attendance(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    full_name = models.CharField(max_length=150)  # This is stored for easier querying/reporting
    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.full_name} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"