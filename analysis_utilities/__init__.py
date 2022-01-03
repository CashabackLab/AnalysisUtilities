def varName(p):
    """
    Return name of the variable as a string
    p: variable of any type
    """
    for k, v in globals().items():
        if id(p) == id(v):
            return k
