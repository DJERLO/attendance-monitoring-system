from django import template

register = template.Library()

@register.filter
def get_item(records, date):
    for record in records:
        if record.date == date:
            return record
    return None