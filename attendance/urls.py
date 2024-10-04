from django.urls import path
from . import views

urlpatterns = [
    path('check/', views.check_attendance, name='check_attendance'),
    path('attendance/', views.attendance, name='attendance'),  # Updated path to use the new attendance view
]