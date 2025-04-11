from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('account/', include('allauth.urls')),  # Include Allauth URLs
    path('account/mfa/', include('allauth.mfa.urls')),  # Ensure this is included
    path('', include('attendance.urls')),
]