""" Format strings for epi package errors. """


def format_type_err_msg(obj, arg_name: str, arg, correct_type) -> str:
    arg_type = arg.__class__
    if arg_type is correct_type:
        raise ValueError("Invalid TypeError message: type(arg) == correct_type.")
    return "%s argument %s must be %s not %s." % (
        obj.__class__.__name__,
        arg_name,
        correct_type.__name__,
        arg_type.__name__,
    )
