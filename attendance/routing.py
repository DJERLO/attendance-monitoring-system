from django.urls import re_path
from attendance.consumers import AttendanceConsumer, NotificationConsumer

websocket_urlpatterns = [
    re_path(r"ws/attendance/$", AttendanceConsumer.as_asgi()),
    re_path(r'ws/notifications/$', NotificationConsumer.as_asgi()), 
]