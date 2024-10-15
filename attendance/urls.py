from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path
from . import views

urlpatterns = [
    path('check/', views.check_attendance, name='check_attendance'),
    path('attendance/', views.attendance, name='attendance'),  # Updated path to use the new attendance view
    path('login/', views.login, name='login'),
    path('register/', views.user_registration, name='employee-registration'),
    path('register-face/<str:employee_number>', views.user_face_registration, name = 'facial-registration'),

     path('online-training/', views.online_training, name='online_training'),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)