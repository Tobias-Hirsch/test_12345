import os
from datetime import datetime
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId # Import ObjectId for MongoDB document IDs
from ..core.config import settings # Global import for configuration settings
from app.schemas.chat_schemas import Attachment # Import the Attachment schema
from app.utils.chat_message_sanitizer import sanitize_conversation_document

# Assuming MONGO_URI and MONGO_DB_NAME are available from environment variables
MONGO_URI = settings.MONGO_URI
MONGO_DB_NAME = settings.MONGO_DB_NAME # Use a dedicated DB for chat

# Reuse the MongoDB client connection logic
def get_mongo_client():
    """Establishes and returns a MongoDB client connection."""
    try:
        # Use the ServerApi version 1
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        # Ping the admin database to confirm a successful connection
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None

# --- Data Structures (using Pydantic schemas for consistency) ---
# We will use the Pydantic schemas defined in chat_schemas.py directly
# instead of defining separate classes here, to ensure consistency.

# --- Service Functions ---

def get_conversations_collection():
    """Gets the MongoDB collection for chat conversations."""
    client = get_mongo_client()
    if client:
        db = client[MONGO_DB_NAME]
        return db["chat_conversations"] # Use a dedicated collection for conversations
    return None

async def create_conversation(user_id: str, title: str) -> Optional[Dict[str, Any]]:
    """Creates a new chat conversation for a user."""
    collection = get_conversations_collection()
    if collection is None:
        return None

    now = datetime.utcnow() # Use UTC time
    conversation_data = {
        "user_id": user_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": []
    }

    try:
        insert_result = collection.insert_one(conversation_data)
        # Retrieve the inserted document to return the full conversation object
        inserted_conversation = collection.find_one({"_id": insert_result.inserted_id})
        if inserted_conversation:
             # Convert ObjectId to string for the response
             inserted_conversation['_id'] = str(inserted_conversation['_id'])
             return inserted_conversation
        return None
    except Exception as e:
        print(f"Error creating conversation: {e}")
        return None

async def get_conversations_by_user(user_id: str) -> List[Dict[str, Any]]:
    """Retrieves all conversations for a given user."""
    collection = get_conversations_collection()
    if collection is None:
        return []

    try:
        # Find conversations for the user, sort by updated_at descending
        conversations_data = list(collection.find({"user_id": user_id}).sort("updated_at", -1))
        # Convert ObjectIds to strings for the response
        for index, conv in enumerate(conversations_data):
            conv = sanitize_conversation_document(conv)
            conversations_data[index] = conv
            conv['_id'] = str(conv['_id'])
            # Also convert message ObjectIds if needed, though typically messages are embedded
            if 'messages' in conv:
                 for msg in conv['messages']:
                     if '_id' in msg:
                         msg['_id'] = str(msg['_id'])
                     if 'attachments' in msg:
                         for att in msg['attachments']:
                             if '_id' in att:
                                 att['_id'] = str(att['_id'])

        return conversations_data
    except Exception as e:
        print(f"Error retrieving conversations for user {user_id}: {e}")
        return []

async def get_all_conversations() -> List[Dict[str, Any]]:
    """Retrieves all chat conversations from the database."""
    collection = get_conversations_collection()
    if collection is None:
        return []

    try:
        conversations_data = list(collection.find({}))
        # Convert ObjectIds to strings for consistency
        for index, conv in enumerate(conversations_data):
            conv = sanitize_conversation_document(conv)
            conversations_data[index] = conv
            conv['_id'] = str(conv['_id'])
            if 'messages' in conv:
                for msg in conv['messages']:
                    if '_id' in msg:
                        msg['_id'] = str(msg['_id'])
                    if 'attachments' in msg:
                        for att in msg['attachments']:
                            if '_id' in att:
                                att['_id'] = str(att['_id'])
        return conversations_data
    except Exception as e:
        print(f"Error retrieving all conversations: {e}")
        return []

async def get_conversation_by_id(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a specific conversation by its ID."""
    collection = get_conversations_collection()
    if collection is None:
        return None

    try:
        # Find the conversation by its ObjectId
        conversation_data = collection.find_one({"_id": ObjectId(conversation_id)})
        if conversation_data:
            conversation_data = sanitize_conversation_document(conversation_data)
            # Convert ObjectId to string for the response
            conversation_data['_id'] = str(conversation_data['_id'])
            if 'messages' in conversation_data:
                 for msg in conversation_data['messages']:
                     if '_id' in msg:
                         msg['_id'] = str(msg['_id'])
                     if 'attachments' in msg:
                         for att in msg['attachments']:
                             if '_id' in att:
                                 att['_id'] = str(att['_id'])
            return conversation_data
        return None
    except Exception as e:
        print(f"Error retrieving conversation {conversation_id}: {e}")
        return None

async def add_message_to_conversation(
    conversation_id: str,
    sender: str,
    content: str,
    attachments: Optional[List[Attachment]] = None,
    search_results: Optional[Dict[str, Any]] = None # Add search_results parameter
) -> bool:
    """Adds a new message (and optional attachments and search results) to a conversation."""
    collection = get_conversations_collection()
    if collection is None:
        return False

    now = datetime.now() # Use UTC time
    new_message_data = {
        "_id": ObjectId(), # Generate ObjectId for the message
        "sender": sender,
        "content": content,
        "timestamp": now,
        "attachments": [att.model_dump(by_alias=True) for att in (attachments if attachments is not None else [])] # Convert Attachment schemas to dicts
    }

    # Add search_results to the message data if provided
    if search_results is not None:
        new_message_data["search_results"] = search_results

    try:
        # Push the new message to the messages array and update updated_at
        update_result = collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {
                "$push": {"messages": new_message_data},
                "$set": {"updated_at": now}
            }
        )
        print(f"Added message to conversation {conversation_id}. Matched: {update_result.matched_count}, Modified: {update_result.modified_count}")
        return update_result.modified_count > 0
    except Exception as e:
        print(f"Error adding message to conversation {conversation_id}: {e}")
        return False

async def delete_attachment_from_conversation(conversation_id: str, message_id: str, attachment_id: str) -> bool:
    """Deletes a specific attachment from a message within a conversation."""
    collection = get_conversations_collection()
    if collection is None:
        return False

    try:
        # Pull the attachment from the specific message's attachments array
        update_result = collection.update_one(
            {"_id": ObjectId(conversation_id), "messages._id": ObjectId(message_id)},
            {
                "$pull": {"messages.$.attachments": {"_id": ObjectId(attachment_id)}},
                "$set": {"updated_at": datetime.utcnow()} # Use UTC time
            }
        )
        print(f"Deleted attachment {attachment_id} from message {message_id} in conversation {conversation_id}. Matched: {update_result.matched_count}, Modified: {update_result.modified_count}")
        return update_result.modified_count > 0
    except Exception as e:
        print(f"Error deleting attachment {attachment_id} from conversation {conversation_id}: {e}")
        return False

async def delete_conversation(conversation_id: str) -> bool:
    """Deletes an entire conversation by its ID."""
    collection = get_conversations_collection()
    if collection is None:
        return False

    try:
        # Delete the conversation document
        delete_result = collection.delete_one({"_id": ObjectId(conversation_id)})
        print(f"Deleted conversation {conversation_id}. Deleted count: {delete_result.deleted_count}")
        if delete_result.deleted_count == 0:
             # Conversation not found or not deleted
             print(f"Warning: Conversation {conversation_id} not found or not deleted.")
             return False # Indicate failure
        return True # Indicate success
    except Exception as e:
        print(f"Error deleting conversation {conversation_id}: {e}")
        raise # Re-raise the exception to be caught by the router

# TODO: Add functions for file size and count limits enforcement,
# potentially during the add_message_to_conversation or a separate validation step.
# This might be better handled on the frontend initially for user feedback,
# but backend validation is also necessary.

async def update_conversation_title(conversation_id: str, new_title: str) -> bool:
    """Updates the title of a specific conversation."""
    collection = get_conversations_collection()
    if collection is None:
        return False

    try:
        update_result = collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {
                "$set": {
                    "title": new_title,
                    "updated_at": datetime.utcnow() # Use UTC time
                }
            }
        )
        print(f"Updated conversation {conversation_id} title to '{new_title}'. Matched: {update_result.matched_count}, Modified: {update_result.modified_count}")
        return update_result.modified_count > 0
    except Exception as e:
        print(f"Error updating conversation {conversation_id} title: {e}")
        return False
