def int_try(val):
    try:
        return int(val)
    except ValueError:
        return False

def bool_try(val):
    if(val=="True" or val=="False"):
        try:
            return bool(val)
        except ValueError:
            return False
    else:
        return False