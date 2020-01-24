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

def format_arch_type_err_msg(arch_type: str) -> str:
    if arch_type == "ar":
        arch_class = "AutoregressiveArch"
    elif arch_type == "coupling":
        arch_class = "CouplingArch"
    else:
        return 'Invalid arch_type "%s".' % arch_type
    return 'Use %s class for arch_type "%s."' % (arch_class, arch_type)
