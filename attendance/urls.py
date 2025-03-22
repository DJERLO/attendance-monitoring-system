from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views
from .api_v1 import api

urlpatterns = [
    # API Endpoints
    path("api/v1/", api.urls), #Include API endpoints

    # Defaults Urls
    path('', views.index, name='index'), # Default page
    path('logout/', views.logout_view, name='logout'), # Logout URL
    
    # Check-In Urls
    path('check-in/', views.check_in_attendance, name='check-in-attendance'), # Check-In URL
    path('attendance/check-in/', views.checking_in, name='attendance-check-in'), # Checking in Endpoint
    
    # Check-Out Urls
    path('check-out/', views.check_out_attendance, name='check-out-attendance'),
    path('attendance/check-out/', views.checking_out, name='attendance-check-out'), # Checking out Endpoint
    
    # Employee Registration URLS
    path('register/', views.user_registration, name='employee-registration'),
    path('register-face/<str:employee_number>', views.user_face_registration, name = 'facial-registration'),
    
    #Face Recognition Test Endpoints
    path('face-verification/', views.face_verification, name='face-verification'),
    path('face-recognition/', views.face_recognition_test, name="face-recognition"),

    # Online-Training for Anti-Spoofing
    path("spoofing-test/", views.upload_image, name="spoofing-test"),
    path('online-training/', views.online_training, name='online_training'),

    # Employee URL's
    path('dashboard/', views.dashboard, name='dashboard'),  # Dashboard URL
    path('attendance-sheet/', views.attendance_sheet, name='attendance-sheet'),  # Default page
    path('attendance-sheet/<int:month>-<int:year>/', views.attendance_sheet_date, name='attendance-sheet-by-date'), #Filter by month and year 
    path('attendance-sheet/<str:employee_number>/<int:month>-<int:year>/', views.employee_attendance_sheet, name='attendance-sheet-by-employee'), #Filter by employee
    path('profile/', views.profile_view, name='profile'),  # Profile URL

    #HR/ADMIN URL's            
]

# Serve media files during development

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)