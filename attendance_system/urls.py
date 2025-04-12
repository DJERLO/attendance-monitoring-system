from django.contrib import admin
from django.shortcuts import redirect
from django.urls import path, include

def admin_login_redirect(request):
    return redirect("account_login")  # allauth's login

urlpatterns = [
    path("admin/login/", admin_login_redirect),  # override admin login
    path('admin/', admin.site.urls),
    path('account/', include('allauth.urls')),  # Include Allauth URLs
    path('account/mfa/', include('allauth.mfa.urls')),  # Ensure this is included
    path('', include('attendance.urls')),
]