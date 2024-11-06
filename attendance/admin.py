from django.contrib import admin
from django.utils.html import mark_safe
from .models import Employee, ShiftRecord
# Register your models here.

# Set custom admin site titles
admin.site.site_header = "Attendance Management System Admin Portal"  # For example, "Attendance System Admin"
admin.site.site_title = "St. Clare College Attendance Management System"    # Appears on the browser tab
admin.site.index_title = "Attendance Management System Admin Portal"  # Appears on the main admin page

class EmployeeAdmin(admin.ModelAdmin):
    list_display = [
        "profile_image_display", 
        "first_name", 
        "middle_name", 
        "last_name", 
        "email", 
        "contact_number", 
        "total_hours_display",  # Add total hours display
        "average_hours_display"  # Add average hours display
    ]
    
    search_fields = ["last_name__startswith", "first_name__startswith", "email"]  # Add search fields here

    def profile_image_display(self, obj):
        if obj.profile_image:
            return mark_safe(f'<img src="{obj.profile_image.url}" width="50" height="50" />')
        return "No Image"
    
    profile_image_display.short_description = "Profile Image"

    def total_hours_display(self, obj):
        return f"{obj.total_hours_worked():.2f}"  # Format to 2 decimal places

    total_hours_display.short_description = "Total Hours Worked"

    def average_hours_display(self, obj):
        return f"{obj.average_hours_worked():.2f}"  # Format to 2 decimal places

    average_hours_display.short_description = "Average Hours Worked"

# Register the Employee model with the EmployeeAdmin
admin.site.register(Employee, EmployeeAdmin)

class ShiftRecordsAdmin(admin.ModelAdmin):
    list_display = ["employee_profile_picture", "employee_full_name", "date", "clock_in_at_am", "clock_out_at_am", "clock_in_at_pm", "clock_out_at_pm", "total_hours"]
    list_select_related = ("employee",)  # Optimizes database queries by fetching related Employee data with ShiftRecord

    def employee_profile_picture(self, obj):
        if obj.employee.avatar_url:
            return mark_safe(f'<img src="{obj.employee.avatar_url}" width="50" height="50" />')
        return "No Image"

    employee_profile_picture.short_description = "Profile Image"

admin.site.register(ShiftRecord, ShiftRecordsAdmin)