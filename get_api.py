import json

def get_api_key_from_json(file_path, api_name):
    """
    Retrieves the API key from a specified JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        str: The API key if found, otherwise None.
    """
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data.get(api_name)  # Updated to match the key with a space
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return None


