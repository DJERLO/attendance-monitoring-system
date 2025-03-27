from django.urls import re_path
from attendance.consumers import AttendanceConsumer

websocket_urlpatterns = [
    re_path(r"ws/attendance/$", AttendanceConsumer.as_asgi()),
]