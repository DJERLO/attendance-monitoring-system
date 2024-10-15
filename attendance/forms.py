from django import forms
from .models import Employee
from django.contrib.auth.models import User

class UserRegistrationForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'password', 'first_name', 'last_name', 'email']
        widgets = {
            'password': forms.PasswordInput(),
        }
    
    def __init__(self, *args, **kwargs):
        super(UserRegistrationForm, self).__init__(*args, **kwargs)
        self.fields['first_name'].required = True
        self.fields['last_name'].required = True
        self.fields['email'].required = True

class EmployeeRegistrationForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = [
            'employee_number',
            'first_name',
            'middle_name',
            'last_name',
            'email',
            'contact_number',
            'profile_image',
        ]
