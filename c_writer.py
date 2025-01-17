"""
Simple module to construct C variables and header files
"""
# Function to convert an array into a C string (requires Numpy)
def create_array(np_array, var_type, var_name, array_dim=1, line_limit=80, indent=4):
    """
    Converts a NumPy array into a C-style array declaration string.
    Returns:
    str: A string containing the C-style array declaration.
    """
    c_str = ""

    # Add array shape
    for i, dim in enumerate(np_array.shape):
        c_str += "const unsigned int " + var_name + "_dim" + str(i + 1) + " = " + str(dim) + ";\n"
    c_str += "\n"

    # Declare C variable
    c_str += "const " + var_type + " " + var_name
    if array_dim == 1:  # 1 dim array
        one_dim_val = 1
        for dim in np_array.shape:
            one_dim_val = one_dim_val * dim
        c_str += "[" + str(one_dim_val) + "]"

    else:
        for dim in np_array.shape:
            c_str += "[" + str(dim) + "]"

    c_str += " = {\n"

    # Create string for the array
    indent = " " * indent
    array_str = indent
    line_len = len(indent)
    val_sep = ", "
    array = np_array.flatten()
    for i, val in enumerate(array):

        # Create a new line if string is over line limit
        val_str = str(val)
        if line_len + len(val_str) + len(val_sep) > line_limit:
            array_str += "\n" + indent
            line_len = len(indent)

        # Add value and separator
        array_str += val_str
        line_len += len(val_str)
        if (i + 1) < len(array):
            array_str += val_sep
            line_len += len(val_sep)

    # Add closing brace
    c_str += array_str + "\n};\n"

    return c_str


# Function to create a header file with given C code as a string
def create_header(c_code, name):
    """
    Generates a C header file content with header guards.
    Args:
        c_code (str): The C code to be included in the header file.
        name (str): The name to be used for the header guard.
    Returns:
        str: The complete C header file content as a string.
    """
    c_str = ""

    # Create header guard
    c_str += "#ifndef " + name.upper() + "_H\n"
    c_str += "#define " + name.upper() + "_H\n\n"

    # Add provided code
    c_str += c_code

    # Close out header guard
    c_str += "\n#endif //" + name.upper() + "_H"

    return c_str
