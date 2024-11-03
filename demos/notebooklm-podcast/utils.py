# -*- coding: utf-8 -*-
# @Time    : 2024/11/3
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : ChatTTSPlus
# @FileName: utils.py

def read_file_to_string(filename):
    # Try UTF-8 first (most common encoding for text files)
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        # If UTF-8 fails, try with other common encodings
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(filename, 'r', encoding=encoding) as file:
                    content = file.read()
                print(f"Successfully read file using {encoding} encoding.")
                return content
            except UnicodeDecodeError:
                continue

        print(f"Error: Could not decode file '{filename}' with any common encoding.")
        return None
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except IOError:
        print(f"Error: Could not read file '{filename}'.")
        return None
