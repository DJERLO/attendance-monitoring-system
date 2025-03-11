from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views
from .api_v1 import api

urlpatterns = [
    # API Endpoints
    path("api/v1/", api.urls), #Include API endpoints

    # Check In Urls
    path('', views.index, name='index'),
    path('check-in/', views.check_in_attendance, name='check-in-attendance'),
    path('attendance/check-in/am/', views.checking_in_at_am, name='attendance-check-in-am'),
    path('attendance/check-in/pm/', views.checking_in_at_pm, name='attendance-check-in-pm'),
    # Check-Out Urls
    path('check-out/', views.check_out_attendance, name='check-out-attendance'),
    path('attendance/check-out/am/', views.checking_out_at_am, name='attendance-check-out-am'),
    path('attendance/check-out/pm/', views.checking_out_at_pm, name='attendance-check-out-pm'),
    
    # Employee Registration URLS
    path('register/', views.user_registration, name='employee-registration'),
    path('register-face/<str:employee_number>', views.user_face_registration, name = 'facial-registration'),
    
    #Face Recognition Test Endpoints
    path('face-verification/', views.face_verification, name='face-verification'),
    path('face-recognition/', views.face_recognition_test, name="face-recognition"),

    # Online-Training for Anti-Spoofing
    path("spoofing-test/", views.upload_image, name="spoofing-test"),
    path('online-training/', views.online_training, name='online_training'),

    # Users URL's
    path('dashboard/', views.dashboard, name='dashboard'),  # Dashboard URL
    path('attendance-sheet/', views.attendance_sheet, name='attendance-sheet'),  # Default page
    path('attendance-sheet/<int:month>-<int:year>/', views.attendance_sheet_date, name='attendance-sheet-by-date'), #Filter by month and year
    path('attendance-sheet/<str:employee_number>/<int:month>-<int:year>/', views.employee_attendance_sheet, name='attendance-sheet-by-employee'),
    path('profile/', views.profile_view, name='profile'),  # Attendance URL            
]

# Serve media files during development

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)