import logging
import sys
from django.apps import AppConfig
from django.db.models.signals import post_migrate
from django.dispatch import receiver

class AttendanceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'attendance'

    def ready(self):
        # Always connect signals, even during migrations
        post_migrate.connect(load_after_migrate, sender=self)

        # Optional: Import for signal connection or module initialization
        try:
            import attendance.signals
        except ImportError as e:
            logging.warning(f"Could not import attendance signals: {e}")

def load_after_migrate(sender, **kwargs):
    from django.contrib.auth.models import Group
    from django.db.utils import OperationalError
    from attendance import recognize_faces

    # Only load these if running app, not during migrations or tests
    if 'runserver' in sys.argv or 'runworker' in sys.argv:
        try:
            # Create default groups
            group_names = [
                'ADMIN',
                'HR ADMIN',
                'TEACHING STAFF (Primary)',
                'TEACHING STAFF (Secondary)',
                'TEACHING STAFF (Tertiary)',
                'NON-TEACHING',
            ]
            for group_name in group_names:
                Group.objects.get_or_create(name=group_name)
            print("Default groups ensured.")

            # Load known faces after migrations
            recognize_faces.load_known_faces()
            print("Known faces loaded.")

        except OperationalError:
            logging.warning("Skipping group creation: database not ready.")
        except Exception as e:
            logging.error(f"Error during post-migrate setup: {e}")
