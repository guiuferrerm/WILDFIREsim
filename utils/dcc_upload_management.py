def read_and_store_dcc_file_at(file_contents, file_name, desired_file_path):
    import base64

    # Decode the uploaded file content from base64
    content_type, content_string = file_contents.split(',')
    decoded = base64.b64decode(content_string)

    # Save the uploaded file to a temporary location on disk
    with open(desired_file_path, 'wb') as f:
        f.write(decoded)
    