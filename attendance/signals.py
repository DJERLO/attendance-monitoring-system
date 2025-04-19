import json
import os
from django.conf import settings
from django.contrib.auth.models import Group
from django.db.models.signals import pre_save, post_save, post_delete
from django.dispatch import receiver
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync, sync_to_async
from django.utils.timezone import localtime

from attendance.recognize_faces import load_known_faces
from .models import Announcement, Employee, FaceImage, LeaveRequest, ShiftRecord, Notification

KNOWN_FACES_DIR = os.path.join(settings.MEDIA_ROOT, 'known_faces')

# Signal receiver to update the face_encoding cache
# when a new Employee instance is created or deleted
@receiver(post_save, sender=FaceImage)
@receiver(post_delete, sender=FaceImage)
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

# Store original status before update
@receiver(pre_save, sender=LeaveRequest)
def cache_request_leave(sender, instance, **kwargs):
    if instance.pk:
        previous = LeaveRequest.objects.get(pk=instance.pk)
        instance._original_status = previous.status
    else:
        instance._original_status = None

@receiver(post_save, sender=LeaveRequest)
def notify_request_leave(sender, instance, created, **kwargs):
    if created:
        # Notify employee that they submitted a request
        Notification.objects.create(
            employee=instance.employee,
            message=f"You filed a leave request for {instance.start_date} to {instance.end_date}. Status: {instance.status}."
        )
        
        # Notify HR here if needed
        # Example: Notify all HR users
        hr_group = Group.objects.get(name="HR ADMIN")
        for hr in hr_group.user_set.all():
            Notification.objects.create(
                employee=hr.employee,  # Assuming linked Employee instance
                message=f"New leave request filed by {instance.employee.full_name()} from {instance.start_date} to {instance.end_date}."
            )

    else:
        if hasattr(instance, '_original_status') and instance._original_status != instance.status:
            if instance.status == "APPROVED":
                status_msg = f"approved by {instance.approved_by.full_name()}" if instance.approved_by else "approved"
            elif instance.status == "REJECTED":
                status_msg = "rejected"
            elif instance.status == "CANCELLED":
                status_msg = "cancelled"
            else:
                status_msg = f"updated to {instance.status.lower()}"

            Notification.objects.create(
                employee=instance.employee,
                message=f"Your leave request from {instance.start_date} to {instance.end_date} was {status_msg}."
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