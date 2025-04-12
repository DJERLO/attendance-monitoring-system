from django import forms
from .models import EmergencyContact, Employee, LeaveRequest
from django.contrib.auth.models import User
from allauth.account.forms import ResetPasswordKeyForm
from django.forms.widgets import TimeInput

class MyCustomResetPasswordKeyForm(ResetPasswordKeyForm):
    def save(self):
        # Add your own processing here (e.g., logging, sending notifications, etc.)
        super(MyCustomResetPasswordKeyForm, self).save()

class UserRegistrationForm(forms.ModelForm):

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name']
    
    def __init__(self, *args, **kwargs):
        super(UserRegistrationForm, self).__init__(*args, **kwargs)
        self.fields['first_name'].required = True
        self.fields['last_name'].required = True

#For Employee Models
class EmployeeRegistrationForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = [
            'employee_number',
            'first_name',
            'middle_name',
            'last_name',
            'gender',
            'birth_date',
            'contact_number',
            'profile_image',
        ]

class EmployeeEmergencyContactForm(forms.ModelForm):
    class Meta:
        model = EmergencyContact
        fields = [
            'contact_name',
            'relationship',
            'phone_number',
            'email',
        ]

#File Leave Form for Employees
class LeaveRequestForm(forms.ModelForm):
    class Meta:
        model = LeaveRequest
        fields = [
            'start_date',
            'end_date',
            'reason',
            'attachment',
        ]
        widgets = {
            'start_date': forms.DateInput(attrs={'type': 'date'}),
            'end_date': forms.DateInput(attrs={'type': 'date'}),
        }   