import json
import os
from django.conf import settings
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync, sync_to_async
from django.utils.timezone import localtime

from attendance.recognize_faces import load_known_faces
from .models import Announcement, Employee, ShiftRecord, Notification

KNOWN_FACES_DIR = os.path.join(settings.MEDIA_ROOT, 'known_faces')

# Signal receiver to update the dashboard
# when a new ShiftRecord is created or updated or when a new Employee is created  
@receiver(post_save, sender=ShiftRecord)
@receiver(post_save, sender=Employee)
def update_dashboard(sender, instance, **kwargs):
    """Trigger WebSocket update when a new ShiftRecord is created or updated."""
    channel_layer = get_channel_layer()
    
    # Count the number of active employees today
    from django.utils.timezone import now
    today = now().date()
    
    #Total number of employees
    total_employees = Employee.objects.all().count()

    # If an Employee instance is saved, get the employee number
    # Determine employee number based on sender
    if sender == Employee:
        employee_number = instance.employee_number  # Directly from Employee model
    elif sender == ShiftRecord:
        employee_number = instance.employee.employee_number  # From related Employee model
    else:
        employee_number = None  # Fallback in case of unexpected sender

    #Employees who have clocked in today
    active_today = ShiftRecord.objects.filter(date=today, status__in=['EARLY', 'PRESENT', 'LATE']).count()
    
    # Send data to WebSocket group
    data = {
        "type": "send_dashboard_update",
        "employee_number": employee_number,
        "total_employees": total_employees,
        "active_today": active_today,
    }

    # Send the message to the WebSocket group
    async_to_sync(channel_layer.group_send)(
        "attendance_group",
        {
            "type": "send_dashboard_update",
            "data": json.dumps(data)
        }
    )

# Signal receiver to update the face_encoding cache
# when a new Employee instance is created or deleted
@receiver(post_save, sender=Employee)
@receiver(post_delete, sender=Employee)
def update_face_cache(sender, instance, **kwargs):
    """Reload face encodings whenever an employee is added or removed"""
    print("Employee face data changed. Reloading known faces...")
    load_known_faces(KNOWN_FACES_DIR)  # Reload faces

# Signal receiver to send real-time notifications
# when a new Notification instance is created
@receiver(post_save, sender=Notification)
def send_notification_realtime(sender, instance, created, **kwargs):
    if created:
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync

        channel_layer = get_channel_layer()
        data = {
            "id": instance.id,
            "message": instance.message,
            "created_at": localtime(instance.created_at).strftime("%b %d, %Y %I:%M %p"),
            "is_read": instance.is_read,
        }

        async_to_sync(channel_layer.group_send)(
            "notifications",
            {
                "type": "send_notification",
                "data": data
            }
        )

@receiver(post_save, sender=ShiftRecord)
def create_clockin_notification(sender, instance, created, **kwargs):
    
    if created and instance.clock_in:
        # Save the notification
        notification = Notification.objects.create(
            employee=instance.employee,
            message=f"You have clocked in at {instance.clock_in.strftime('%I:%M %p on %B %d, %Y')}."
        )
    
    if created and instance.clock_out:
        # Save the notification
        notification = Notification.objects.create(
            employee=instance.employee,
            message=f"You have clocked out at {instance.clock_out.strftime('%I:%M %p on %B %d, %Y')}."
        )

@receiver(post_save, sender=Announcement)
def create_notifications_for_announcement(sender, instance, created, **kwargs):
    if created:
        employees = Employee.objects.exclude(id=instance.created_by.id) if instance.created_by else Employee.objects.all()
        notifications = [
            Notification(
                employee=employee,
                message=f"New Announcement: {instance.title}"
            )
            for employee in employees
        ]
        Notification.objects.bulk_create(notifications)