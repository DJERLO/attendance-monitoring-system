from django import forms
from .models import Employee

class EmployeeRegistrationForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = [
            'employee_id',
            'first_name',
            'middle_name',
            'last_name',
            'email',
            'contact_number',
            'profile_image',
        ]
