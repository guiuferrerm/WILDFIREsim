def convert_seconds_to_dhms(total_seconds):
    days = total_seconds // 86400
    remaining = total_seconds % 86400

    hours = remaining // 3600
    remaining %= 3600

    minutes = remaining // 60
    seconds = remaining % 60

    return int(days), int(hours), int(minutes), int(seconds)