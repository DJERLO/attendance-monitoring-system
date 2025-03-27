import json
from django.db.models.signals import post_save
from django.dispatch import receiver
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .models import Employee, ShiftRecord

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
    employee_number = instance.employee.employee_number  # Adjust based on your model

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