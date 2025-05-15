import csv
from django.contrib import admin
from django.db.models import Q
from django.http import HttpResponse
from django.utils.html import mark_safe
from django.db.models import Value, CharField
from django.db.models.functions import Concat
from django.contrib.auth.models import User, Group, Permission
from django.contrib.auth.admin import UserAdmin as DefaultUserAdmin
from django.urls import reverse
from django.utils.html import format_html
from django.contrib import messages
from rangefilter.filters import (DateRangeFilterBuilder)

from .models import Announcement, Camera, EmergencyContact, Employee, FaceImage, LeaveRequest, Notification, ShiftRecord, WorkHours, Event

# Set custom admin site titles
admin.site.site_header = "Attendance Management System Admin Portal"  # For example, "Attendance System Admin"
admin.site.site_title = "St. Clare College Attendance Management System"    # Appears on the browser tab
admin.site.index_title = "Attendance Monitoring System Admin Portal"  # Appears on the main admin page

class WorkHoursAdmin(admin.ModelAdmin):
    list_display = ["open_time", "close_time", "id"]  # Add a non-editable field first
    list_display_links = ["id"]  # Make "id" the clickable link
    readonly_fields = ['id']  # Makes 'id' field read-only
    list_editable = ["open_time", "close_time"]  # Now these are editable in the list view
    list_per_page = 1  # Ensures only one record is shown per page

    def has_add_permission(self, request):
        """Prevents adding a new WorkHours instance if one already exists."""
        return not WorkHours.objects.exists()  # Disable "Add" if record exists

class UserInline(admin.StackedInline):  # or TabularInline for table layout
    model = User
    extra = 0  # Don't show extra blank forms
    can_delete = False  # Prevent removing it
    verbose_name_plural = "User Account Info"
    fields = (
        'username', 'password', 'first_name', 'last_name',
        'email', 'is_active', 'is_superuser', 'is_staff',
    )

# Inline Emergency Contact model
class EmergencyContactInline(admin.StackedInline):  # or TabularInline for table layout
    model = EmergencyContact
    extra = 0  # Don't show extra blank forms
    can_delete = False  # Prevent removing it
    verbose_name_plural = "Emergency Contact Info"


# Employee Inline (including image preview)
class EmployeeInline(admin.StackedInline):
    model = Employee
    extra = 0
    show_change_link = True
    readonly_fields = ['profile_image_preview']
    verbose_name_plural = "Employee Information"
    fields = (
        'employee_number', 'first_name', 'middle_name', 'last_name',
        'gender', 'birth_date', 'hire_date',
        'email', 'contact_number', 'group',
        'employment_status', 'hourly_rate', 'profile_image', 'profile_image_preview'
    )

    def profile_image_preview(self, obj):
        if obj.profile_image:
            return mark_safe(
                f'<img src="{obj.profile_image.url}" width="100" height="100" style="object-fit: cover; border-radius: 8px;" />'
            )
        return "No image uploaded"
    profile_image_preview.short_description = "Profile Preview"

    # Dynamically include or exclude the 'group' field based on the user’s group
    def get_fields(self, request, obj=None):
        fields = list(super().get_fields(request, obj))  # Get the default fields
        if not request.user.groups.filter(name='ADMIN').exists():
            if 'group' in fields:
                fields.remove('group')  # Remove the 'group' field for non-admin users
        return fields

class EmployeeAdmin(admin.ModelAdmin):
    inlines = [EmergencyContactInline]
     # Fields that can be searched in the admin panel
    search_fields = ["first_name__icontains", "last_name__icontains", "email__icontains", "employee_number__icontains"]

    # Fields to filter on in the admin panel
    list_filter = ["user__is_active", "employee_number", "employment_status", "group"]

    # Pagination settings for list view
    list_per_page = 20  # Limit number of records per page

    readonly_fields = ('profile_image_preview',)

    # List of fields displayed in the admin list view
    list_display = [
        "profile_image_display", 
        "full_name", 
        "group",
        'employment_status', 
        "total_hours_display",  
        "average_hours_display"
    ]

    list_display_links = ["profile_image_display"]  # Make "profile_image_display" clickable

    # Restrict Non-Admin users from editing the group field in the list view
    def get_list_editable(self, request):
        if request.user.groups.filter(name='ADMIN').exists():
            return ['group', 'employment_status']
        if request.user.groups.filter(name__icontains='HR').exists():
            return ['employment_status']
        return []

    # Override the changelist_view to dynamically set list_editable
    def changelist_view(self, request, extra_context=None):
        self.list_editable = self.get_list_editable(request)
        return super().changelist_view(request, extra_context)

    # Fieldsets
    fieldsets = (
        ('Account Information', {
            'fields': ('profile_image_preview', 'profile_image')
        }),
        ('Personal Info', {
            'fields': ('first_name', 'middle_name', 'last_name', 'gender', 'birth_date'),
        }),
        ('Contact Info', {
            'fields': ('email', 'contact_number'),
        }),
        ('Employment Details', {
            'fields': ('employee_number', 'hire_date', 'group', 'employment_status', 'hourly_rate'),
        }),
    )

    
    # Restrict Group Field in the Edit Form, Only ADMIN can edit
    def get_readonly_fields(self, request, obj=None):
        readonly = list(super().get_readonly_fields(request, obj))
        if not request.user.groups.filter(name='ADMIN').exists():
            readonly.append('group')
        return readonly
    
    # Restrict HR users from adding employees directly in the Employee Admin
    def has_add_permission(self, request):
        # HR ADMIN and ADMIN users cannot add employees directly in the EmployeeAdmin
        if request.user.groups.filter(name__icontains='ADMIN').exists():
            return False
        return super().has_add_permission(request)
    

    # Making the profile image clickable for easier access
    def profile_image_display(self, obj):
        if obj.profile_image:
            change_url = reverse('admin:%s_%s_change' % (obj._meta.app_label, obj._meta.model_name), args=[obj.pk])
            return format_html('<a href="{0}"><img src="{1}" width="100" height="100" /></a>', change_url, obj.profile_image.url)
        return "No Image"

    profile_image_display.short_description = "Profile Image"

    def profile_image_preview(self, obj):
        if obj.profile_image:
            return mark_safe(
                f'<img src="{obj.profile_image.url}" width="100" height="100" style="object-fit: cover; border-radius: 8px;" />'
            )
        return "No image uploaded"
    profile_image_preview.short_description = "Current Profile Image"

    # Displaying the total hours worked by an employee
    def total_hours_display(self, obj):
        return f"{obj.total_hours_worked():.2f}"

    total_hours_display.short_description = "Total Hours Worked"

    # Displaying the average hours worked by an employee
    def average_hours_display(self, obj):
        return f"{obj.average_hours_worked():.2f}"

    average_hours_display.short_description = "Average Hours Worked"

    # Action to set the selected employees as active
    def set_active(self, request, queryset):
        queryset.update(user__is_active=True)
    set_active.short_description = "Set selected employees as Active"

    # Action to set the selected employees as inactive
    def set_inactive(self, request, queryset):
        queryset.update(user__is_active=False)
    set_inactive.short_description = "Set selected employees as Inactive"

    # Add the actions to the admin panel
    actions = [set_active, set_inactive]

# Register the Employee model with the EmployeeAdmin
admin.site.register(Employee, EmployeeAdmin)

class ShiftRecordsAdmin(admin.ModelAdmin):
    list_display = ["employee_profile_picture", "employee_full_name", "date", "clock_in", "clock_out", "total_hours"]
    list_select_related = ("employee",)  # Optimizes database queries by fetching related Employee data with ShiftRecord
    list_filter = [
        ("date", admin.DateFieldListFilter),  # Date range filter for the date field
        "employee__group",  # Filter by employee group
        "employee__hourly_rate",  # Filter by whether the employee has an hourly rate
    ]

    def save_model(self, request, obj, form, change):
        from attendance.models import LeaveRequest

        has_approved_leave = LeaveRequest.objects.filter(
            employee=obj.employee,
            status='APPROVED',
            start_date__lte=obj.date,
            end_date__gte=obj.date
        ).exists()

        if has_approved_leave:
            messages.warning(request, f"{obj.employee.full_name()} has an approved leave on this day.")

        super().save_model(request, obj, form, change)

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
            "Full Name",
            "Date",
            "Clock-In",
            "Clock-Out",
            "Total Hours",
        ])

        # Write data rows
        for record in queryset:
            writer.writerow([
                record.employee.full_name(),
                record.date,
                record.clock_in,
                record.clock_out,
                record.total_hours,
            ])

        return response

    # Configure the admin action
    export_to_csv.short_description = "Export Selected Records to CSV"
    actions = [export_to_csv]

admin.site.register(ShiftRecord, ShiftRecordsAdmin)
admin.site.register(WorkHours, WorkHoursAdmin)

@admin.register(FaceImage)
class FaceImageAdmin(admin.ModelAdmin):
    list_display = ('employee', 'image_preview', 'uploaded_at')
    search_fields = ('employee__name', 'employee__employee_id')
    list_filter = ('uploaded_at',)

    def image_preview(self, obj):
        """Displays an image preview in Django Admin."""
        if obj.image:
            return format_html('<img src="{}" width="100" style="border-radius: 5px;">', obj.image.url)
        return "(No Image)"

    image_preview.short_description = "Face Preview"

class EventsAdmin(admin.ModelAdmin):
    list_display = ('title', 'get_days_of_week', 'start', 'end')
    list_filter = (
        ('start', DateRangeFilterBuilder()),  # Date range filter for the start date
        'all_day',  # Filter by whether the event is all-day or not
    )
    search_fields = ('title', 'description')  # Search by title and description
    ordering = ('start',)  # Default ordering by event start date

    def get_days_of_week(self, obj):
        day_map = {
            '0': 'Sun',
            '1': 'Mon',
            '2': 'Tue',
            '3': 'Wed',
            '4': 'Thu',
            '5': 'Fri',
            '6': 'Sat'
        }
        return ", ".join([day_map[day] for day in obj.days_of_week]) if obj.days_of_week else "N/A"
    
    get_days_of_week.short_description = "Repeats On"  # Display name in admin


# Register the Event model with the custom EventsAdmin 
admin.site.register(Event, EventsAdmin)
admin.site.register(Notification)
from django.utils.html import mark_safe
class CameraAdmin(admin.ModelAdmin):
    list_display = ("name", "mode", "camera_url", "is_active", 'snapshot')
    list_filter = ("mode", "is_active")
    search_fields = ("name", "camera_url")
    list_editable = ['mode' , 'is_active']

    def snapshot(self, obj):
        return mark_safe(f'<img src="/snapshot/{obj.id}/" width="300" height="200" />')

    snapshot.short_description = 'Thumbnail'

admin.site.register(Camera, CameraAdmin)

class LeaveRequestAdmin(admin.ModelAdmin):
    list_display = ("employee", "start_date", "end_date", "status", "duration", "approved_by", "created_at")
    list_filter = ("status", "start_date", "end_date")
    search_fields = ("employee__first_name", "employee__last_name", "employee__employee_number")
    date_hierarchy = "start_date"
    autocomplete_fields = ["employee", "approved_by"]
    list_editable = ['status']
    readonly_fields = ("created_at", "updated_at")

    def get_fieldsets(self, request, obj=None):
        fieldsets = [
            ("Leave Details", {
                "fields": ("employee", "start_date", "end_date", "reason", "attachment")
            }),
            ("Status Information", {
                "fields": ("status",),
            }),
            ("Timestamps", {
                "fields": ("created_at", "updated_at"),
                "classes": ("collapse",),
            }),
        ]

        # Only show 'approved_by' to superusers
        if request.user.is_superuser:
            fieldsets[1][1]["fields"] += ("approved_by",)
        return fieldsets

    def get_readonly_fields(self, request, obj=None):
        readonly = list(super().get_readonly_fields(request, obj))
        # Non-superusers should not be able to edit 'approved_by'
        if not request.user.is_superuser:
            readonly.append('approved_by')
        return readonly
   

    def save_model(self, request, obj, form, change):
        """
        Override the save_model method to ensure that the `approved_by` field is automatically populated
        with the currently logged-in user's Employee object when the leave request status is set to 'APPROVED'.
        
        This is necessary because the Django admin interface does not pass the `request` context to the model's 
        `save()` method by default, and we need to track who approved the leave request.
        
        Parameters:
            request (HttpRequest): The request object containing the logged-in user.
            obj (LeaveRequest): The leave request object being saved.
            form (ModelForm): The form being used to validate the data.
            change (bool): A boolean indicating whether this is an existing object being updated (True) or a new object (False).
        """
        if obj.status == 'APPROVED' and not obj.approved_by:
            obj.approved_by = request.user.employee
        obj.save()

admin.site.register(LeaveRequest, LeaveRequestAdmin)

class AnnouncementAdmin(admin.ModelAdmin):
    list_display = ("title", "created_by", "created_at", "is_active")
    list_filter = ("is_active", "created_at")
    search_fields = ("title", "message")
    autocomplete_fields = ["created_by"]

    # Exclude the created_by field from the admin form
    exclude = ('created_by',)

    def save_model(self, request, obj, form, change):
        """
        Override the save_model method to automatically set the `created_by` field 
        to the currently logged-in user when creating a new announcement.

        This ensures that the `created_by` field is properly populated with the user who created 
        the announcement, as the Django admin interface doesn't automatically set this field.

        Parameters:
            request (HttpRequest): The request object containing the logged-in user.
            obj (Announcement): The announcement object being saved.
            form (ModelForm): The form being used to validate the data.
            change (bool): A boolean indicating whether this is an existing object being updated (True) or a new object (False).
        """
        if not obj.created_by:
            # Set the `created_by` field to the currently logged-in user's Employee object
            obj.created_by = request.user.employee
        
        # Save the updated announcement object
        obj.save()

admin.site.register(Announcement, AnnouncementAdmin)

# Unregister the original User admin
admin.site.unregister(User)

# Re-register with our custom inline
@admin.register(User)
class CustomUserAdmin(DefaultUserAdmin):
    inlines = [EmployeeInline]

    # Exclude superusers from the list view
    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        if not request.user.is_superuser:
            # Exclude superusers for non-superusers
            queryset = queryset.exclude(is_superuser=True)
        return queryset

    # Make 'is_superuser' read-only for non-superusers
    def get_readonly_fields(self, request, obj=None):
        readonly_fields = super().get_readonly_fields(request, obj)
        if obj and not obj.is_superuser:
            # If the user is not a superuser, make 'is_superuser' read-only
            readonly_fields = readonly_fields + ('is_superuser',)
        return readonly_fields

    # Optionally, if you want to prevent them from changing their own user details (like the 'is_superuser' field):
    def has_change_permission(self, request, obj=None):
        if obj and obj.is_superuser:
            return False  # Prevent superusers from editing their own details
        return super().has_change_permission(request, obj)

    # Optionally, if you want to restrict the 'is_superuser' field from being changed at all:
    def formfield_for_dbfield(self, db_field, request, **kwargs):
        if db_field.name == "is_superuser":
            if request.user.is_superuser:
                kwargs['disabled'] = True  # Disable the is_superuser field
        return super().formfield_for_dbfield(db_field, request, **kwargs)
    
    # Exclude specific fields from the admin form
    fieldsets = (
        (None, {
            'fields': (
                'username', 
                'first_name', 
                'last_name', 
                'email', 
                'password', 
                # Add any other fields you want to keep here
            ),
        }),
        ('Status', {
            'fields': (
                'is_active', 
                'is_superuser',  # Optional: Leave it as readonly if you want it visible but non-editable
            ),
        }),
    )
    
    # Exclude 'is_staff', 'groups', and 'user_permissions' for non-superusers
    def get_fieldsets(self, request, obj=None):
        fieldsets = super().get_fieldsets(request, obj)
        
        # If the user is a superuser, include the excluded fields in the form
        if request.user.is_superuser:
            # Add 'is_staff', 'groups', and 'user_permissions' back for superusers
            fieldsets = list(fieldsets)
            fieldsets.append(('Permissions', {
                'fields': ('is_staff', 'groups', 'user_permissions')
            }))
        
        return fieldsets
    
class GroupAdmin(admin.ModelAdmin):
    list_display = ['name']
    search_fields = ('name', 'permissions__name')
    list_filter = ('permissions',)
    filter_horizontal = ('permissions',)
    ordering = ('name',)

# Unregister the default Group admin
admin.site.unregister(Group)

# Register the new customized Group admin
admin.site.register(Group, GroupAdmin)