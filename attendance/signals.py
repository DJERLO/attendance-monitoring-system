from datetime import timedelta
import json
import os
from django.conf import settings
from django.contrib.auth.models import Group
from django.db.models.signals import pre_save, post_save, post_delete
from django.dispatch import receiver
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from django.template.loader import render_to_string
from django.utils.timezone import localtime
from django.utils import timezone
from django.core.cache import cache
from attendance.recognize_faces import load_known_faces
from attendance.views import get_status_summary_per_day, get_top_employees
from .models import Announcement, Employee, FaceImage, LeaveRequest, ShiftRecord, Notification

KNOWN_FACES_DIR = os.path.join(settings.MEDIA_ROOT, 'known_faces')

# Signal receiver to update the face_encoding cache
# when a new Employee instance is created or deleted
@receiver(post_save, sender=FaceImage)
@receiver(post_delete, sender=FaceImage)
def update_face_cache(sender, instance, **kwargs):
    """Reload face encodings whenever an employee is added or removed"""
    print("Employee face data changed. Reloading known faces...")
    cache.delete('known_faces')
    load_known_faces()  # Reload faces

# Signal receiver to send real-time notifications
# when a new Notification instance is created
@receiver(post_save, sender=Notification)
def send_notification_realtime(sender, instance, created, **kwargs):
    if created:
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync

        # Get the employee associated with the notification
        employee = instance.employee
        channel_layer = get_channel_layer()
        
        data = {
            "id": instance.id,
            "message": instance.message,
            "created_at": localtime(instance.created_at).strftime("%b %d, %Y %I:%M %p"),
            "is_read": instance.is_read,
        }

        async_to_sync(channel_layer.group_send)(
            f"notifications_{employee.id}",
            {
                "type": "send_notification",
                "data": data
            }
        )

@receiver(post_save, sender=ShiftRecord)
def create_clockin_notification(sender, instance, created, **kwargs):
    
    if created and instance.clock_in and instance.clock_out:
        # Send notification to that employee when attendance is completed
        local_clock_in = timezone.localtime(instance.clock_in)
        local_clock_out = timezone.localtime(instance.clock_out)
        notification = Notification.objects.create(
            employee=instance.employee,
            message=f"Your attendance for {local_clock_in.strftime('%b %d, %Y %I:%M %p')} to {local_clock_out.strftime('%b %d, %Y %I:%M %p')} is recorded.",
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
                employee = hr.employee,
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
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync
        from django.utils.timezone import localtime

        employees = Employee.objects.exclude(id=instance.created_by.id) if instance.created_by else Employee.objects.all()
        channel_layer = get_channel_layer()
        notifications = []

        for employee in employees:
            notification = Notification.objects.create(
                employee=employee,
                message=f"New Announcement: {instance.title}"
            )

            # Send WebSocket notification to each employee's group
            data = {
                "id": notification.id,
                "message": notification.message,
                "created_at": localtime(notification.created_at).strftime("%b %d, %Y %I:%M %p"),
                "is_read": notification.is_read,
            }

            async_to_sync(channel_layer.group_send)(
                f"notifications_{employee.id}",
                {
                    "type": "send_notification",
                    "data": data
                }
            )

# Dashboard Signals
@receiver(post_save, sender=Employee)
@receiver(post_delete, sender=Employee)
@receiver(post_save, sender=ShiftRecord)
@receiver(post_delete, sender=ShiftRecord) #For attendance updates (Forgot to add this)
def dashboard_update(sender, instance, **kwargs):
    today = timezone.now().date()
    
    employees = Employee.objects.all()
    total_employees = employees.count()
    top_10_employees = get_top_employees()

    # Render the top 10 leaderboard HTML fragment (server-side rendering)
    leaderboard_html = render_to_string("user/components/charts/leaderboard.html", {
        "top_employees": top_10_employees
    })

    today_summary = get_status_summary_per_day(today)
    active_today = today_summary["EARLY"] + today_summary["PRESENT"] + today_summary["LATE"]

    # Date range for the current month
    first_day_of_month = today.replace(day=1)
    last_day_of_month = (first_day_of_month + timedelta(days=31)).replace(day=1) - timedelta(days=1)
    all_dates = [first_day_of_month + timedelta(days=i) for i in range((last_day_of_month - first_day_of_month).days + 1)]
   
    attendance_data = {
        "labels": [date.strftime("%Y-%m-%d") for date in all_dates],
        "EARLY": [],
        "PRESENT": [],
        "LATE": [],
        "ABSENT": [],
    }

    for date in all_dates:
        summary = get_status_summary_per_day(date)
        for status in attendance_data:
            if status != "labels":
                attendance_data[status].append(summary.get(status, 0))

    # Attendance overview (today)
    absent_count = total_employees - active_today
    attendance_overview = {
        "EARLY": today_summary["EARLY"],
        "PRESENT": today_summary["PRESENT"],
        "LATE": today_summary["LATE"],
        "ABSENT": absent_count,
    }

    #Latest Shift(Attendance)
    shiftlogs = ShiftRecord.objects.all().order_by('-date')

    # Render the top 10 leaderboard HTML fragment (server-side rendering)
    attendancelog_html = render_to_string("user/components/attendance_logs.html", {
        "shiftlogs": shiftlogs
    })

    channel_layer = get_channel_layer()

    data = {
        "attendance_data": attendance_data,
        "attendance_overview": attendance_overview,
        "employees": {
            "total_employees": employees.count(),
            "active_employees": active_today,
        },
        "leaderboard": leaderboard_html, 
        "shiftlog": attendancelog_html,
    }

    async_to_sync(channel_layer.group_send)(
            "attendance_group",
            {
                "type": "attendance_update",
                "data": data
            }
        )
