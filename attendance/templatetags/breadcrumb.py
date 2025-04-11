from django import template

register = template.Library()

@register.filter
def breadcrumb_path(value):
    """Splits the request path and returns a list of breadcrumb parts."""
    parts = value.strip('/').split('/')
    breadcrumbs = []
    url = ''

    for part in parts:
        url += f'/{part}'
        breadcrumbs.append({'name': part.replace('-', ' ').title(), 'url': url})

    return breadcrumbs