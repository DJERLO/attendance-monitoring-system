import sys
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views
from .api_v1 import api
from allauth.account.views import PasswordChangeView
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
    path('account/', views.profile_view, name='profile'),  # Profile URL
    path('account/change-password/', PasswordChangeView.as_view(), name='account_change_password'),
    path('account/manage-face-images/', views.manage_face_images, name='Manage-Face'),
    path('delete-face-image/<int:image_id>/', views.delete_face_image, name='delete_face_image'),
    path('calendar', views.school_event_calendar, name='event-calendar'),
    path("create-event/", views.create_event, name="create_event"),
    path('update-profile/', views.update_profile, name='update-profile'),
    path('file-leave/', views.request_leave_view, name='file-leave'),
    
    #HR/ADMIN URL's
    path('employees/', views.employee_management, name='employee-list'),  # Employee List URL
    path('employees/<str:employee_number>/', views.employee_details, name='employee-details'),  # Employee Detail URL
    path('announcements/', views.announcement_board, name='announcement_board'),

    # Notifications URL's
    path('notifications/mark-read/<int:id>/', views.mark_notification_read, name='mark_notification_read'),
    path('notifications/mark-all-read/', views.mark_all_notifications_read, name='mark_all_notifications_read'),
    path('notifications/delete/<int:id>/', views.delete_notification, name='delete_notification'),

]

# Serve media files during development
from django.urls import re_path
from django.views.static import serve
# Automatically serve media files only when using `runserver`
if settings.MEDIA_ROOT:
    urlpatterns += [
        re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
    ]