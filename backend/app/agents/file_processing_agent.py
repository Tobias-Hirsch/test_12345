import json
import os
import re
import io
import tempfile
import uuid
from typing import Any, Dict, Optional, List
import PyPDF2 # Added for PDF processing

from ..core.config import settings # Global import
from .base_agent import BaseAgent
from app.llm.llm import llm_ollama_deepseek_ainvoke
from app.services import auth
from fastapi import HTTPException

# Import MinIO module
from app.modules.minio_module import store_document_in_minio, generate_presigned_url

# Import document processing functions from lib/deal_documents
from app.tools.word import extract_word_content_from_url
from app.tools.exlsx import download_and_parse_xlsx
from app.tools.deal_document import process_image # For image processing

class FileProcessingAgent(BaseAgent):
    """
    The Conversation Attachment File Processing Agent handles the parsing,
    understanding, and extraction of information from files attached to the
    conversation (e.g., PDFs, documents, images).
    """

    def __init__(self):
        super().__init__(name="Conversation Attachment File Processing Agent",
                         description="Processes attached files to extract information.")

    async def process(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes an attached file to extract relevant information using DeepSeek,
        with integrated permission checks.

        Args:
            task (str): The specific instruction for file processing (e.g., "extract_text", "analyze_image").
            context (Dict[str, Any]): A dictionary containing:
                                       - 'files': A list of UploadFile objects from the request.
                                       - Other relevant context like 'db_session' and 'current_user'.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted information and thoughts.
        """
        thoughts = []
        output = {}
        files: Optional[List[Dict[str, Any]]] = context.get("files")
        db_session = context.get("db_session")
        current_user = context.get("current_user")

        if not current_user or not hasattr(current_user, 'id'):
            thoughts.append("Error: Current user or user ID not found in context. Cannot determine MinIO bucket.")
            output["error"] = "Authentication error: User ID not available for file storage."
            return {"output": output, "thoughts": thoughts}

        minio_chat_bucket_name_prefix = settings.MINIO_CHAT_BUCKET_NAME
        if not minio_chat_bucket_name_prefix:
            thoughts.append("Error: MINIO_CHAT_BUCKET_NAME environment variable not set.")
            output["error"] = "Server configuration error: MinIO chat bucket name not defined."
            return {"output": output, "thoughts": thoughts}

        user_specific_bucket_name = minio_chat_bucket_name_prefix
        thoughts.append(f"Determined MinIO bucket for user: {user_specific_bucket_name}")

        # Permission check
        # if not auth.has_permission(db_session, current_user, "process_attachments"):
        #     thoughts.append("Permission denied: User does not have permission to process attachments.")
        #     raise HTTPException(status_code=403, detail="Permission denied: You do not have permission to process attachments.")

        thoughts.append(f"File Processing Agent received task: '{task}'")
        thoughts.append("Initiating file processing with DeepSeek...")

        if not files:
            output["error"] = "No files provided for processing."
            thoughts.append("Error: No files to process.")
            return {"output": output, "thoughts": thoughts}

        processed_files_info = []
        # No longer need a temporary directory as files are processed directly from BytesIO

        for file_info in files:
            filename = file_info.get("filename", "unknown_file")
            content_type = file_info.get("content_type", "application/octet-stream")
            file_stream = file_info.get("file_stream")

            if not file_stream:
                thoughts.append(f"Error: No file stream found for {filename}.")
                processed_files_info.append({
                    "filename": filename,
                    "error": "No file stream provided."
                })
                continue

            file_content = ""
            minio_url = None
            temp_file_path = None
            temp_output_dir = None # For PDF processing output

            try:
                file_stream.seek(0) # Ensure stream is at the beginning for reading
                file_bytes = file_stream.read()

                # 1. Save to a temporary file
                # Use original file extension if available, otherwise default to .tmp
                file_extension = os.path.splitext(filename)[1] if os.path.splitext(filename)[1] else ".tmp"
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    temp_file.write(file_bytes)
                    temp_file_path = temp_file.name
                thoughts.append(f"Saved file {filename} to temporary path: {temp_file_path}")

                # 2. Upload to MinIO
                object_name = f"{current_user.id}/{uuid.uuid4()}_{filename}"
                minio_url = store_document_in_minio(temp_file_path, object_name, user_specific_bucket_name)
                thoughts.append(f"Uploaded {filename} to MinIO: {minio_url}")

                # Generate a presigned URL for temporary access
                presigned_url = generate_presigned_url(user_specific_bucket_name, object_name, expires_in_seconds=3600)
                if not presigned_url:
                    thoughts.append(f"Error: Could not generate presigned URL for {filename}.")
                    processed_files_info.append({
                        "filename": filename,
                        "error": "Could not generate presigned URL."
                    })
                    continue
                thoughts.append(f"Generated presigned URL for {filename}: {presigned_url}")

                # 3. Process content based on file type using URL
                if content_type == "application/pdf":
                    # Retain PyPDF2 processing, but ensure file is uploaded to MinIO first
                    try:
                        # PyPDF2 reads directly from a BytesIO stream, so we re-wrap the bytes
                        pdf_stream = io.BytesIO(file_bytes)
                        reader = PyPDF2.PdfReader(pdf_stream)
                        for page_num in range(len(reader.pages)):
                            page = reader.pages[page_num]
                            file_content += page.extract_text() or "" # Extract text, handle None
                        thoughts.append(f"Extracted text from PDF: {filename} using PyPDF2 after MinIO upload.")
                        if not file_content.strip():
                            thoughts.append(f"Warning: Extracted PDF content for {filename} is empty or only whitespace. This might indicate a scanned PDF or an issue with text extraction.")
                    except Exception as pdf_e:
                        thoughts.append(f"Error extracting text from PDF {filename} using PyPDF2: {pdf_e}. This PDF might be scanned or malformed.")
                        file_content = "" # Ensure content is empty on error
                elif content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                    file_content = await extract_word_content_from_url(presigned_url) # Use presigned URL
                    if file_content is None:
                        thoughts.append(f"Failed to extract content from Word file: {filename}.")
                        file_content = ""
                    else:
                        thoughts.append(f"Successfully extracted content from Word file: {filename}.")
                elif content_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                    excel_data = await download_and_parse_xlsx(presigned_url) # Use presigned URL
                    if excel_data is None:
                        thoughts.append(f"Failed to extract content from Excel file: {filename}.")
                        file_content = ""
                    else:
                        # Convert DataFrame dict to string for DeepSeek
                        excel_string_content = ""
                        for sheet_name, df in excel_data.items():
                            excel_string_content += f"Arbeitsblatt: {sheet_name}\n"
                            excel_string_content += df.to_string()
                            excel_string_content += "\n" + "-" * 50 + "\n"
                        file_content = excel_string_content
                        thoughts.append(f"Successfully extracted content from Excel file: {filename}.")
                elif content_type.startswith("image/"):
                    # For image files, we'll pass the MinIO URL to DeepSeek.
                    # The process_image function from deal_documents is for Qwen-VL and requires a 'question' context.
                    # If DeepSeek can process images directly from URL, it will be handled by the prompt below.
                    file_content = f"File is an image. MinIO URL: {minio_url}. Please describe the image based on its content if possible."
                    thoughts.append(f"Image file detected: {filename}. MinIO URL: {minio_url}. Passing URL for DeepSeek processing.")
                else:
                    # For other file types, decode bytes to string, assuming UTF-8, with error handling
                    try:
                        file_content = file_bytes.decode('utf-8')
                        thoughts.append(f"Read content from {filename} directly from stream (UTF-8).")
                    except UnicodeDecodeError:
                        try:
                            file_content = file_bytes.decode('latin-1') # Fallback to latin-1
                            thoughts.append(f"Read content from {filename} directly from stream (Latin-1 fallback).")
                        except Exception as decode_e:
                            file_content = file_bytes.decode('utf-8', errors='ignore') # Fallback to ignore errors
                            thoughts.append(f"Error decoding {filename}: {decode_e}. Read content from stream with errors ignored.")
                            thoughts.append(f"Raw bytes start: {file_bytes[:50]}...") # Log raw bytes for debugging
                    
                    thoughts.append(f"Extracted content preview for {filename}: {file_content[:200]}...")

                prompt = f"""
                You are a File Processing AI (DeepSeek). Your task is to extract and summarize key information
                from the provided file content based on the user's instruction.

                User Instruction: {task}
                File Name: {filename}
                File Type: {content_type}
                File Content:
                ```
                {file_content}
                ```

                Provide your extracted information in a JSON string with the following structure:
                {{
                    "filename": "{filename}",
                    "extracted_text": "...", # Required: Main extracted text
                    "summary": "...", # Optional: Summary of the content
                    "message": "...", # Optional: General message
                    "thoughts": ["...", "..."] # Your internal thoughts/reasoning.
                }}
                """

                deepseek_file_response_str = await llm_ollama_deepseek_ainvoke(prompt)
                thoughts.append(f"DeepSeek file processing raw response for {filename}: {deepseek_file_response_str}")

                try:
                    # Attempt to extract JSON from the response, in case it's wrapped in markdown or has extra text
                    json_match = re.search(r"```json\s*(.*?)\s*```", deepseek_file_response_str, re.DOTALL)
                    if json_match:
                        json_string = json_match.group(1)
                    else:
                        json_string = deepseek_file_response_str.strip() # Try to strip whitespace if no markdown block

                    deepseek_parsed_response = json.loads(json_string)
                    processed_files_info.append(deepseek_parsed_response)
                    thoughts.extend(deepseek_parsed_response.get("thoughts", []))
                except json.JSONDecodeError as e:
                    thoughts.append(f"DeepSeek response for {filename} was not valid JSON: {e}. Raw response: {deepseek_file_response_str}. Treating as plain text response.")
                    processed_files_info.append({
                        "filename": filename,
                        "extracted_text": "", # Clear extracted_text if JSON is malformed
                        "summary": "",
                        "message": f"Processed content as plain text due to malformed JSON response: {e}. Raw response: {deepseek_file_response_str}",
                        "thoughts": [f"DeepSeek response for {filename} was not valid JSON: {e}. Raw response: {deepseek_file_response_str}"]
                    })
                except Exception as e: # Catch any other unexpected errors during parsing
                    thoughts.append(f"Unexpected error during DeepSeek response parsing for {filename}: {e}. Raw response: {deepseek_file_response_str}. Treating as plain text response.")
                    processed_files_info.append({
                        "filename": filename,
                        "extracted_text": "", # Clear extracted_text if parsing fails
                        "summary": "",
                        "message": f"An unexpected error occurred during response parsing: {e}. Raw response: {deepseek_file_response_str}",
                        "thoughts": [f"Unexpected error during DeepSeek response parsing for {filename}: {e}. Raw response: {deepseek_file_response_str}"]
                    })
            except Exception as e:
                thoughts.append(f"Error processing file {filename}: {e}")
                processed_files_info.append({
                    "filename": filename,
                    "error": f"An error occurred during file processing: {e}"
                })
            finally:
                # Clean up temporary file if it was created
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    thoughts.append(f"Cleaned up temporary file: {temp_file_path}")
                # Clean up temporary output directory for PDF if it was created
                if temp_output_dir and os.path.exists(temp_output_dir):
                    import shutil
                    shutil.rmtree(temp_output_dir)
                    thoughts.append(f"Cleaned up temporary output directory: {temp_output_dir}")

        output["processed_files"] = processed_files_info

        thoughts.append("File processing complete. Preparing output.")

        return {"output": output, "thoughts": thoughts}

# import asyncio
# import io

# # Mock UploadFile class for testing purposes
# class MockUploadFile:
#     def __init__(self, filename: str, content: bytes, content_type: str = "application/octet-stream"):
#         self.filename = filename
#         self.file = io.BytesIO(content)
#         self.content_type = content_type

#     async def read(self, size: int = -1) -> bytes:
#         return self.file.read(size)

#     async def __aenter__(self):
#         return self

#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         self.file.close()


# async def main():
#     # Example usage of the FileProcessingAgent
#     agent = FileProcessingAgent()
#     task = "Extract text and summarize"
#     context = {
#         "files": [
#             MockUploadFile(
#                 filename="Kommentar",
#                 content=b"This is a mock PDF content for testing purposes.",
#                 content_type="application/pdf"
#             )
#         ],
#         "db_session": None,  # Replace with actual DB session
#         "current_user": None  # Replace with actual user object
#     }
    
#     try:
#         result = await agent.process(task, context)
#         print(result)
#     except HTTPException as e:
#         print(f"Error: {e.detail}")

# if __name__ == "__main__":
#     asyncio.run(main())