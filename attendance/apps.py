import sys
from django.apps import AppConfig
from django.db.models.signals import post_migrate
from django.dispatch import receiver

class AttendanceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'attendance'

    def ready(self):
        import attendance.signals  # Ensure signals are loaded
        from attendance import recognize_faces

       # Only load faces during normal app startup
        if 'runserver' in sys.argv or 'runworker' in sys.argv:
            try:
                # Connect the signal to load faces after migrations
                post_migrate.connect(load_faces_after_migrate, sender=self)

            except Exception as e:
                print(f"Could not set up signals for loading known faces: {e}")

def load_faces_after_migrate(sender, **kwargs):
    from attendance import recognize_faces
    try:
        recognize_faces.load_known_faces()
    except Exception as e:
        print(f"Could not load known faces after migration: {e}")
