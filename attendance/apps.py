import sys
from django.apps import AppConfig


class AttendanceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'attendance'

    def ready(self):
        import attendance.signals  # Ensure signals are loaded
        from attendance import recognize_faces

        # Only load faces during normal app startup
        if 'runserver' in sys.argv or 'runworker' in sys.argv:
            try:
                recognize_faces.load_known_faces()
            except Exception as e:
                print(f"Could not load known faces during startup: {e}")
