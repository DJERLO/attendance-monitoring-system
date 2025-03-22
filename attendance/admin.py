import csv
from django.contrib import admin
from django.db.models import Q
from django.http import HttpResponse
from django.utils.html import mark_safe
from datetime import datetime
from django.db.models import Value, CharField
from django.db.models.functions import Concat
from rangefilter.filters import (
    DateRangeFilterBuilder,
    DateTimeRangeFilterBuilder,
    NumericRangeFilterBuilder,
    DateRangeQuickSelectListFilterBuilder,
)
from .models import Employee, ShiftRecord, WorkHours

# Set custom admin site titles
admin.site.site_header = "Attendance Management System Admin Portal"  # For example, "Attendance System Admin"
admin.site.site_title = "St. Clare College Attendance Management System"    # Appears on the browser tab
admin.site.index_title = "Attendance Management System Admin Portal"  # Appears on the main admin page

class WorkHoursAdmin(admin.ModelAdmin):
    list_display = ["open_time", "close_time"]

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
    list_display = ["employee_profile_picture", "employee_full_name", "date", "clock_in", "clock_out", "total_hours"]
    list_select_related = ("employee",)  # Optimizes database queries by fetching related Employee data with ShiftRecord
    list_filter = [("date", DateRangeFilterBuilder())]
    
    def get_queryset(self, request):
        """
        Annotates employee full name dynamically for search.
        """
        queryset = super().get_queryset(request).annotate(
            employee_full_name=Concat('employee__first_name', Value(' '), 'employee__middle_name', Value(' '), 'employee__last_name', output_field=CharField())
        )
        return queryset

    def get_search_results(self, request, queryset, search_term):
        if not search_term:
            return queryset, False

        # Dynamically filter on annotated `employee_full_name`
        queryset = queryset.filter(employee_full_name__icontains=search_term)
        return queryset, True

    # Register search fields
    search_fields = ["employee__full_name", "employee__employee_id"]
    
    def employee_profile_picture(self, obj):
        if obj.employee.avatar_url:
            return mark_safe(f'<img src="{obj.employee.avatar_url}" width="50" height="50" />')
        return "No Image"

    employee_profile_picture.short_description = "Profile Image"

    def export_to_csv(self, request, queryset):
        """
        Exports selected shift records into a CSV with a dynamically generated filename
        depending on whether all selected records belong to one employee.
        """
        # Check if all selected records are for the same employee
        employees = queryset.values_list('employee', flat=True).distinct()
        
        if employees.count() == 1:
            # Only one unique employee is selected
            employee_name = queryset.first().employee.full_name().replace(" ", "_")
            file_name = f"shift_records_{employee_name}.csv"
        else:
            # Multiple employees are selected
            file_name = "shift_records.csv"

        # Create response
        response = HttpResponse(content_type="text/csv")
        response['Content-Disposition'] = f'attachment; filename="{file_name}"'

        # Write to CSV
        writer = csv.writer(response)
        writer.writerow([
            "Employee Full Name",
            "Date",
            "Clock-In (AM)",
            "Clock-Out (AM)",
            "Clock-In (PM)",
            "Clock-Out (PM)",
            "Total Hours",
        ])

        # Write data rows
        for record in queryset:
            writer.writerow([
                record.employee.full_name(),
                record.date,
                record.clock_in_at_am,
                record.clock_out_at_am,
                record.clock_in_at_pm,
                record.clock_out_at_pm,
                record.total_hours,
            ])

        return response

    # Configure the admin action
    export_to_csv.short_description = "Export Selected Records to CSV"
    actions = [export_to_csv]

admin.site.register(ShiftRecord, ShiftRecordsAdmin)
admin.site.register(WorkHours, WorkHoursAdmin)