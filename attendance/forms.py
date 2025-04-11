from django import forms
from .models import EmergencyContact, Employee
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
