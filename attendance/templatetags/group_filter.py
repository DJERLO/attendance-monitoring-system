from django import template

register = template.Library()

@register.filter(name='has_group')
def has_group(user, group_name):
    """
    Custom template filter to check if a user belongs to a specific group.
    Usage in templates: {% if user|has_group:"HR" %}
    """
    return user.groups.filter(name=group_name).exists()