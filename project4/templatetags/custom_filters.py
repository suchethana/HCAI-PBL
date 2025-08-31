from django import template

register = template.Library()

@register.filter
def replace(value, arg):
    if isinstance(value, str) and isinstance(arg, str):
        parts = arg.split('|', 1)
        if len(parts) == 2:
            old, new = parts
            return value.replace(old, new)
    return value