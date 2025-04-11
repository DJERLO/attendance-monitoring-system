from django import template

register = template.Library()

@register.filter
def get_item(records, date):
    # Collect all records that match the date
    matching_records = [record for record in records if record.date == date]
    return matching_records
