def build_npz_file(arrays):
    import numpy as np
    import io
    # Save to in-memory buffer
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    buffer.seek(0)
    file_data = buffer.getvalue()

    return file_data

def read_and_store_npz_contents(file_path):
    import numpy as np
    # Load the .npz file
    data = np.load(file_path, allow_pickle=True)
    
    # Create an empty dictionary
    data_dict = {}
    
    # Store the arrays in the dictionary
    for key in data.files:
        data_dict[key] = data[key]

    return data_dict
