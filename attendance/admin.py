from django.contrib import admin
from .models import Employee
# Register your models here.
class EmployeeAdmin(admin.ModelAdmin):
    list_display = ["profile_image", "first_name", "middle_name", "last_name", "email", "contact_number"]
 
admin.site.register(Employee, EmployeeAdmin)