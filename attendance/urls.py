from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('check/', views.check_attendance, name='check_attendance'),
    path('attendance/', views.attendance, name='attendance'),  # Updated path to use the new attendance view
    path('register/', views.employee_registration, name='employee-registration'),
    path('register-face/<str:employee_id>', views.upload_face_images, name = 'facial-registration'),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)