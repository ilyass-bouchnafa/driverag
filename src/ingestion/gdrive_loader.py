"""
gdrive_loader.py
-----------------
Utilities to authenticate with Google Drive and to list/download
supported files from a designated Drive folder. This module centralizes
the Drive OAuth flow and provides helpers used by the ingestion
and synchronization pipeline.

Notes for contributors:
- Add your Google API client secret to `credentials/credentials.json`.
- The OAuth flow will create `credentials/token.pickle` on first run.
- Do NOT commit credential files to the repository.
"""

import io
import logging
import os
import pickle

# Used to send HTTP requests when refreshing expired OAuth tokens
from google.auth.transport.requests import Request

# Handles the OAuth2 authorization flow for installed applications
from google_auth_oauthlib.flow import InstalledAppFlow

# Used to construct a resource object for interacting with Google APIs
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

from googleapiclient.http import MediaIoBaseDownload

# OAuth scopes define the permissions requested from the user. Use
# 'https://www.googleapis.com/auth/drive' for full read/write access.
# For read-only use 'https://www.googleapis.com/auth/drive.readonly'.
SCOPES = ['https://www.googleapis.com/auth/drive']

# ====================== FIXED PATHS (current structure) ======================
import os
from pathlib import Path

# Compute the project root path (C:\driverag)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Paths to local credential files used by the OAuth flow.
# Keep these files out of version control and share them securely.
CREDENTIALS_FILE = str(ROOT_DIR / "credentials" / "credentials.json")
TOKEN_FILE = str(ROOT_DIR / "credentials" / "token.pickle")

logger.info(f"Credentials path: {CREDENTIALS_FILE}")
logger.info(f"Token path: {TOKEN_FILE}")
# ================================================================================

def get_drive_service():
    """
    Authenticate the application with Google Drive using OAuth2 and
    return a Google Drive API service object.

    Authentication process:
    -----------------------
    1. If a saved token exists locally, it is loaded.
    2. If the token is expired but has a refresh token, it is refreshed automatically.
    3. If no valid token exists, the OAuth flow is triggered:
        - A browser window opens
        - The user logs into their Google account
        - The user grants permission to the application
    4. The new token is saved locally for future executions.

    Returns
    -------
    googleapiclient.discovery.Resource
        A Google Drive service object that allows interaction
        with the Google Drive API (listing files, downloading documents, etc.).
    """

    # Initialize credentials variable
    # It will later contain OAuth access credentials
    creds = None

    # ---------------------------------------------------------
    # STEP 1: Check if a previously saved authentication token exists
    # ---------------------------------------------------------
    if os.path.exists(TOKEN_FILE):

        # Open the token file in binary read mode
        with open(TOKEN_FILE, 'rb') as token:

            # Deserialize the stored credentials object using pickle
            # This restores the OAuth credentials from the previous session
            creds = pickle.load(token)

    # ---------------------------------------------------------
    # STEP 2: Validate the credentials
    # ---------------------------------------------------------
    # If credentials do not exist or are no longer valid,
    # we must refresh them or start a new authentication flow
    if not creds or not creds.valid:

        # Case 1: Credentials exist but have expired
        # If a refresh token is available, we can request a new access token
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.warning(
                    "Google Drive refresh token failed; deleting stale token and reauthenticating: %s",
                    e
                )
                creds = None
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE)

        # Case 2: No credentials exist (first run)
        # or the credentials cannot be refreshed
        if not creds or not creds.valid:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"Google credentials file not found: {CREDENTIALS_FILE}. "
                    "Place your client secrets there."
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES
            )
            creds = flow.run_local_server(port=0)

        # ---------------------------------------------------------
        # STEP 3: Save the new credentials for future executions
        # ---------------------------------------------------------
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    
    # ---------------------------------------------------------
    # STEP 4: Build the Google Drive API service object
    # ---------------------------------------------------------
    # This object acts as a client interface for interacting with
    # Google Drive (listing files, downloading documents, etc.)
    service = build('drive', 'v3', credentials=creds)

    # Return the authenticated Drive API service
    return service


def list_files_recursive(folder_id: str, path: str = "") -> list[dict]:
    """
    Recursively list all supported files inside a Google Drive folder
    and its subfolders.

    The function explores the folder structure and builds the full
    path of each file (for example: "Courses/Semester2/TP3.pdf").
    Only files with supported MIME types are kept.

    Parameters
    ----------
    folder_id : str
        The Google Drive folder ID to explore.

    path : str
        The current folder path used to reconstruct the full file path.
        It is empty at the first call.

    Returns
    -------
    list[dict]
        A list of dictionaries containing metadata for each supported file:
        - id
        - name
        - mimeType
        - path (full folder path)
        - format
        - modifiedTime
    """

    # Import the dictionary containing supported MIME types
    from src.config import SUPPORTED_MIME_TYPES

    # Initialize the Google Drive API service (authenticated client)
    service = get_drive_service()

    # -----------------------------------------------------------
    # Build a query to retrieve all items inside the given folder
    # -----------------------------------------------------------
    # The query means:
    # - the parent folder must match folder_id
    # - the item must not be in the trash
    # Note: This will return both files and sub-folders.
    query = f"'{folder_id}' in parents and trashed=false"

    # -----------------------------------------------------------
    # Send the request to the Google Drive API
    # -----------------------------------------------------------
    # .list() : Prepares the search with the defined filters
    # .execute() : Sends the request and returns the data as a dictionary
    results = service.files().list(
        q=query,  # Apply the query filter (parent ID and not trashed)
        fields="files(id, name, mimeType, modifiedTime)",  # Fetch only the fields we need (efficient)
        orderBy="name"  # Tell Google to sort the list by name before sending it
    ).execute()

    # Extract the items from the response, defaulting to an empty list if no files are found
    items = results.get('files', [])

    # This list will store all discovered files
    all_files = []

    # -----------------------------------------------------------
    # Iterate through each item in the folder
    # -----------------------------------------------------------
    for item in items:

        # Build the full path of the item
        # If path already exists -> append the file name
        # Example: "Courses/AI/slides.pdf"
        item_path = f"{path}/{item['name']}" if path else item['name']

        # -------------------------------------------------------
        # Case 1: The item is a folder
        # -------------------------------------------------------
        if item['mimeType'] == 'application/vnd.google-apps.folder':

            # Log the folder being explored (useful for debugging)
            logger.info(f"Exploring folder: {item_path}")

            # Recursively explore the subfolder
            sub_files = list_files_recursive(item['id'], item_path)

            # Add all discovered files from the subfolder
            all_files.extend(sub_files)

        # -------------------------------------------------------
        # Case 2: The item is a supported file
        # -------------------------------------------------------
        elif item['mimeType'] in SUPPORTED_MIME_TYPES:

            # Add the reconstructed full path to the metadata
            item['path'] = item_path

            # Convert the MIME type into a simpler format label
            # Example: "application/pdf" -> "pdf"
            item['format'] = SUPPORTED_MIME_TYPES[item['mimeType']]

            # Add modifiedTime for synchronization
            item['modifiedTime'] = item.get('modifiedTime', '')

            # Store the file metadata in the result list
            all_files.append(item)

        # -------------------------------------------------------
        # Case 3: Unsupported file format
        # -------------------------------------------------------
        else:
            # Unsupported file types are intentionally skipped
            logger.warning(
                f"Unsupported format ignored: {item['name']} ({item['mimeType']})"
            )

    # Return the list of all supported files discovered in the folder tree        
    return all_files


# ======================================================================================================
# from src.config import GOOGLE_DRIVE_FOLDER_ID

# if __name__ == "__main__":
    
#     FOLDER_ID = GOOGLE_DRIVE_FOLDER_ID  # Replace with your folder ID

#     # Call the function to list files recursively
#     files = list_files_recursive(FOLDER_ID)

#     # Print the results
#     for file in files:
#         print(f"Found file: {file['path']} (ID: {file['id']}, Format: {file['format']})")
# ======================================================================================================

def download_file(file_id: str, file_name: str, mime_type: str) -> bytes:
    """
    Download a file from Google Drive directly into memory using BytesIO.

    Special case:
    -------------
    Native Google Docs files cannot be downloaded directly as binary files.
    They must first be exported into a compatible format such as DOCX.

    Parameters
    ----------
    file_id : str
        The unique Google Drive file ID.

    file_name : str
        The file name (used only for logs and tracking).

    mime_type : str
        The MIME type of the file, used to determine
        whether export is required.

    Returns
    -------
    bytes
        The binary content of the downloaded file.
    """

    # Initialize the authenticated Google Drive API service
    # This service allows access to file download endpoints
    service = get_drive_service()

    # Log the current file being downloaded (helps when running sync)
    logger.info(f"⬇️  Downloading file: {file_name}")
    

    # ---------------------------------------------------------
    # STEP 1: Determine the correct download method
    # ---------------------------------------------------------
    # Google native documents (Google Docs) cannot be downloaded
    # with get_media() because they do not exist as binary files.
    # They must first be exported to a standard format.
    if mime_type == 'application/vnd.google-apps.document':

        # Export the Google Doc into DOCX format
        request = service.files().export_media(
            fileId = file_id,
            mimeType = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    
    else:
        # Standard files can be downloaded directly
        request = service.files().get_media(fileId=file_id)
    
    # ---------------------------------------------------------
    # STEP 2: Create an in-memory buffer
    # ---------------------------------------------------------
    # BytesIO creates a temporary memory buffer that behaves like a file
    # without writing anything to disk
    buffer = io.BytesIO()

    # Create the downloader object that transfers the file
    # chunk by chunk into the buffer
    downloader = MediaIoBaseDownload(buffer, request)

    # ---------------------------------------------------------
    # STEP 3: Download the file progressively
    # ---------------------------------------------------------
    # Google API sends large files in chunks.
    # The loop continues until the download is fully completed.
    done = False

    while not done:

        # next_chunk() downloads the next available chunk
        # It returns progress information and completion status
        _, done = downloader.next_chunk()
    
    # ---------------------------------------------------------
    # STEP 4: Reset buffer position
    # ---------------------------------------------------------
    # After writing, the cursor is at the end of the buffer.
    # seek(0) moves it back to the beginning before reading.
    buffer.seek(0)

    # Return the full binary content of the file
    return buffer.read()



    