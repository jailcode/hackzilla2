import uuid
from datetime import datetime
from db import container  # Assuming container is already set up to connect to Cosmos DB

def save_user_note(user_id, title, content):
    """
    Save a new note for a user in the database.

    Args:
        user_id (str): The ID of the user.
        title (str): The title of the note.
        content (str): The content of the note.

    Returns:
        dict: The created note object.
    """
    note = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "title": title,
        "content": content,
        "created_at": datetime.utcnow().isoformat()
    }
    container.create_item(body=note)
    return note

def get_all_notes():
    """
    Retrieve all notes from the database.

    Returns:
        list: A list of all notes.
    """
    query = "SELECT * FROM c"  # Query to get all notes
    notes = list(container.query_items(query=query, enable_cross_partition_query=True))
    return notes