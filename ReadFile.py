def read_file(file_path):
    """
    Reads the contents of a text file and returns it as a string.

    Parameters:
    file_path (str): The path to the text file.

    Returns:
    str: The contents of the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        return data
    except FileNotFoundError:
        return f"Error: The file '{file_path}' was not found."
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage:
# file_content = read_file('example.txt')
# print(file_content)

