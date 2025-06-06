import os
import shutil
import pathlib
import logging
import paramiko # Ensure this is uncommented or added

# Configure logger for this module
logger = logging.getLogger(__name__)

class FileShuttle:
    """
    A utility class for performing local and remote (SCP) file operations.
    """
    def __init__(self):
        """
        Initializes the FileShuttle utility.
        """
        logger.info("FileShuttle initialized.")

    # --- Local Operations ---
    def create_directory(self, path: str, exist_ok: bool = True, parents: bool = True) -> bool:
        logger.info(f"Attempting to create directory: {path} (exist_ok={exist_ok}, parents={parents})")
        try:
            pathlib.Path(path).mkdir(parents=parents, exist_ok=exist_ok)
            logger.info(f"Successfully created directory: {path}")
            return True
        except FileExistsError as e:
            logger.error(f"Directory already exists and exist_ok=False: {path} - {e}")
            return False
        except PermissionError as e:
            logger.error(f"Permission denied when creating directory: {path} - {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}", exc_info=True)
            return False

    def local_exists(self, path: str) -> bool:
        logger.debug(f"Checking existence of path: {path}")
        return pathlib.Path(path).exists()

    def local_list(self, path: str) -> list[str]:
        logger.info(f"Listing contents of directory: {path}")
        if not self.local_exists(path):
            logger.error(f"Directory not found for listing: {path}")
            return []
        if not pathlib.Path(path).is_dir():
            logger.error(f"Path is not a directory, cannot list contents: {path}")
            return []
        try:
            return [p.name for p in pathlib.Path(path).iterdir()]
        except PermissionError as e:
            logger.error(f"Permission denied when listing directory: {path} - {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to list directory {path}: {e}", exc_info=True)
            return []

    def local_delete(self, path_str: str) -> bool:
        logger.info(f"Attempting to delete path: {path_str}")
        path_obj = pathlib.Path(path_str)
        try:
            if not path_obj.exists():
                logger.warning(f"Path not found for deletion: {path_str}")
                return False

            if path_obj.is_file() or path_obj.is_symlink():
                path_obj.unlink()
                logger.info(f"Successfully deleted file/symlink: {path_str}")
            elif path_obj.is_dir():
                shutil.rmtree(path_obj)
                logger.info(f"Successfully deleted directory: {path_str}")
            else:
                logger.warning(f"Path is not a file or directory, cannot delete: {path_str}")
                return False
            return True
        except PermissionError as e:
            logger.error(f"Permission denied when deleting path: {path_str} - {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete path {path_str}: {e}", exc_info=True)
            return False

    def local_copy(self, source_path_str: str, dest_path_str: str, overwrite: bool = False) -> bool:
        logger.info(f"Attempting to copy from '{source_path_str}' to '{dest_path_str}' (overwrite={overwrite})")
        source_path = pathlib.Path(source_path_str)
        dest_path = pathlib.Path(dest_path_str)

        try:
            if not source_path.exists():
                logger.error(f"Source path not found for copy: {source_path_str}")
                return False

            actual_dest_path = dest_path
            if dest_path.is_dir() and source_path.is_file(): # Copying file into directory
                 actual_dest_path = dest_path / source_path.name
            elif dest_path.is_dir() and source_path.is_dir(): # Copying dir into dir (actual_dest_path is the new dir name inside dest_path)
                 actual_dest_path = dest_path / source_path.name


            if actual_dest_path.exists() and not overwrite:
                logger.error(f"Destination path '{actual_dest_path}' already exists and overwrite is False.")
                return False

            if actual_dest_path.exists() and overwrite:
                logger.info(f"Overwrite is True, removing existing destination: {actual_dest_path}")
                if not self.local_delete(str(actual_dest_path)):
                    logger.error(f"Failed to delete existing destination '{actual_dest_path}' for overwrite.")
                    return False

            if not actual_dest_path.parent.exists():
                 self.create_directory(str(actual_dest_path.parent), exist_ok=True, parents=True)

            if source_path.is_dir():
                shutil.copytree(source_path, actual_dest_path, dirs_exist_ok=overwrite)
                logger.info(f"Successfully copied directory from '{source_path}' to '{actual_dest_path}'")
            elif source_path.is_file():
                shutil.copy2(source_path, actual_dest_path)
                logger.info(f"Successfully copied file from '{source_path}' to '{actual_dest_path}'")
            else:
                logger.warning(f"Source path '{source_path}' is not a file or directory. Cannot copy.")
                return False
            return True

        except shutil.Error as e:
            logger.error(f"Shutil error during copy from '{source_path_str}' to '{dest_path_str}': {e}", exc_info=True)
            return False
        except PermissionError as e:
            logger.error(f"Permission denied during copy: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to copy from '{source_path_str}' to '{dest_path_str}': {e}", exc_info=True)
            return False

    def local_move(self, source_path_str: str, dest_path_str: str, overwrite: bool = False) -> bool:
        logger.info(f"Attempting to move from '{source_path_str}' to '{dest_path_str}' (overwrite={overwrite})")
        source_path = pathlib.Path(source_path_str)
        dest_path = pathlib.Path(dest_path_str)

        try:
            if not source_path.exists():
                logger.error(f"Source path not found for move: {source_path_str}")
                return False

            actual_dest_target_path = dest_path
            if dest_path.is_dir(): # If dest is an existing directory, move source inside it
                actual_dest_target_path = dest_path / source_path.name

            if actual_dest_target_path.exists() and overwrite:
                logger.info(f"Overwrite is True, removing existing destination: {actual_dest_target_path}")
                if not self.local_delete(str(actual_dest_target_path)):
                    logger.error(f"Failed to delete existing destination '{actual_dest_target_path}' for overwrite.")
                    return False
            elif actual_dest_target_path.exists() and not overwrite:
                logger.error(f"Destination '{actual_dest_target_path}' already exists and overwrite is False.")
                return False

            if not actual_dest_target_path.parent.exists():
                self.create_directory(str(actual_dest_target_path.parent), exist_ok=True, parents=True)

            shutil.move(str(source_path), str(actual_dest_target_path))
            logger.info(f"Successfully moved from '{source_path_str}' to '{actual_dest_target_path}'")
            return True
        except shutil.Error as e:
             logger.error(f"Shutil error during move from '{source_path_str}' to '{dest_path_str}': {e}", exc_info=True)
             return False
        except PermissionError as e:
            logger.error(f"Permission denied during move: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to move from '{source_path_str}' to '{dest_path_str}': {e}", exc_info=True)
            return False

    # --- SCP Operations ---
    def _create_ssh_client(self, hostname: str, port: int, username: str, password=None, key_filepath=None) -> paramiko.SSHClient | None:
        logger.info(f"Creating SSH client for {username}@{hostname}:{port}")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            if key_filepath:
                key_filepath_abs = str(pathlib.Path(key_filepath).expanduser().resolve())
                logger.info(f"Attempting SSH connection with key: {key_filepath_abs}")
                client.connect(hostname, port=port, username=username, key_filename=key_filepath_abs, timeout=10)
            elif password:
                logger.info("Attempting SSH connection with password.")
                client.connect(hostname, port=port, username=username, password=password, timeout=10)
            else:
                logger.error("SSH connection requires either a password or a key_filepath.")
                client.close()
                return None
            logger.info(f"SSH connection established to {username}@{hostname}:{port}")
            return client
        except paramiko.AuthenticationException as e:
            logger.error(f"SSH Authentication failed for {username}@{hostname}: {e}")
        except paramiko.SSHException as e:
            logger.error(f"SSH connection error for {username}@{hostname}: {e}", exc_info=True)
        except FileNotFoundError: # This is for the key_filepath
            logger.error(f"SSH key file not found: {key_filepath}")
        except TimeoutError: # Python's built-in TimeoutError for client.connect timeout
             logger.error(f"SSH connection timed out for {username}@{hostname}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during SSH connection to {username}@{hostname}: {e}", exc_info=True)

        client.close() # Ensure client is closed on failure path before returning None
        return None

    def _exec_remote_command(self, client: paramiko.SSHClient, command: str) -> bool:
        logger.info(f"Executing remote command: {command}")
        try:
            stdin, stdout, stderr = client.exec_command(command, timeout=10)
            exit_status = stdout.channel.recv_exit_status() # Wait for command to complete
            if exit_status == 0:
                logger.info(f"Remote command executed successfully: {command}")
                # logger.debug(f"Stdout: {stdout.read().decode()}") # Optional: log stdout
                return True
            else:
                logger.error(f"Remote command failed with exit status {exit_status}: {command}")
                error_output = stderr.read().decode()
                if error_output: # Only log if there's actual error output
                    logger.error(f"Stderr: {error_output}")
                return False
        except paramiko.SSHException as e:
            logger.error(f"Failed to execute remote command '{command}': {e}", exc_info=True)
            return False
        except TimeoutError: # Python's built-in TimeoutError from exec_command timeout
            logger.error(f"Timeout executing remote command: {command}")
            return False
        except Exception as e: # Catch any other unexpected error
            logger.error(f"Unexpected error executing remote command '{command}': {e}", exc_info=True)
            return False


    def scp_upload_file(self, local_path_str: str, remote_path_str: str, hostname: str, username: str, password=None, key_filepath=None, port=22) -> bool:
        logger.info(f"Attempting SCP upload of file '{local_path_str}' to '{username}@{hostname}:{remote_path_str}'")
        local_path = pathlib.Path(local_path_str)
        if not local_path.is_file():
            logger.error(f"Local path for SCP upload is not a file: {local_path_str}")
            return False

        client = self._create_ssh_client(hostname, port, username, password, key_filepath)
        if not client:
            return False

        try:
            remote_parent_dir = str(pathlib.Path(remote_path_str).parent)
            if not self._exec_remote_command(client, f'mkdir -p "{remote_parent_dir}"'):
                 logger.error(f"Failed to create remote parent directory '{remote_parent_dir}'. Upload aborted.")
                 return False

            with client.open_scp() as scp:
                scp.put(str(local_path), remote_path_str)
            logger.info(f"File '{local_path_str}' uploaded successfully to '{remote_path_str}'")
            return True
        except paramiko.SCPException as e: # paramiko.SCPException is from scp.py, not a base paramiko exception
            logger.error(f"SCP error during file upload: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error during SCP file upload: {e}", exc_info=True)
            return False
        finally:
            if client:
                client.close()
                logger.debug("SSH client closed after scp_upload_file.")

    def scp_download_file(self, remote_path_str: str, local_path_str: str, hostname: str, username: str, password=None, key_filepath=None, port=22) -> bool:
        logger.info(f"Attempting SCP download of file '{remote_path_str}' from '{username}@{hostname}' to '{local_path_str}'")
        local_path_obj = pathlib.Path(local_path_str)

        if not self.create_directory(str(local_path_obj.parent), exist_ok=True, parents=True):
            logger.error(f"Failed to create local parent directory '{local_path_obj.parent}'. Download aborted.")
            return False

        actual_local_path = local_path_str
        if local_path_obj.is_dir(): # If specified local_path is a dir, download into it with remote filename
            actual_local_path = str(local_path_obj / pathlib.Path(remote_path_str).name)
            logger.debug(f"Local path is a directory. Effective local path for file: {actual_local_path}")

        client = self._create_ssh_client(hostname, port, username, password, key_filepath)
        if not client:
            return False

        try:
            with client.open_scp() as scp:
                scp.get(remote_path_str, actual_local_path)
            logger.info(f"File '{remote_path_str}' downloaded successfully to '{actual_local_path}'")
            return True
        except paramiko.SCPException as e:
            logger.error(f"SCP error during file download: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error during SCP file download: {e}", exc_info=True)
            return False
        finally:
            if client:
                client.close()
                logger.debug("SSH client closed after scp_download_file.")

    def scp_upload_directory(self, local_dir_str: str, remote_dir_str: str, hostname: str, username: str, password=None, key_filepath=None, port=22) -> bool:
        logger.info(f"Attempting SCP upload of directory '{local_dir_str}' to '{username}@{hostname}:{remote_dir_str}'")
        local_dir_path = pathlib.Path(local_dir_str)
        if not local_dir_path.is_dir():
            logger.error(f"Local path for SCP upload is not a directory: {local_dir_str}")
            return False

        client = self._create_ssh_client(hostname, port, username, password, key_filepath)
        if not client:
            return False

        try:
            # Ensure base remote directory exists, scp.put recursive might not create it if it's multi-level
            if not self._exec_remote_command(client, f'mkdir -p "{remote_dir_str}"'):
                 logger.error(f"Failed to create base remote directory '{remote_dir_str}'. Upload aborted.")
                 return False

            with client.open_scp() as scp:
                scp.put(str(local_dir_path), remote_path=remote_dir_str, recursive=True)
            logger.info(f"Directory '{local_dir_str}' uploaded successfully to '{remote_dir_str}'")
            return True
        except paramiko.SCPException as e:
            logger.error(f"SCP error during directory upload: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error during SCP directory upload: {e}", exc_info=True)
            return False
        finally:
            if client:
                client.close()
                logger.debug("SSH client closed after scp_upload_directory.")

    def scp_download_directory(self, remote_dir_str: str, local_dir_str: str, hostname: str, username: str, password=None, key_filepath=None, port=22) -> bool:
        logger.info(f"Attempting SCP download of directory '{remote_dir_str}' from '{username}@{hostname}' to '{local_dir_str}'")

        # Ensure local base directory exists or can be created.
        # scp.get with recursive=True will place the downloaded directory *inside* local_dir_str.
        # e.g., scp.get('/remote/foo', '/local', recursive=True) results in /local/foo
        self.create_directory(local_dir_str, exist_ok=True, parents=True)

        client = self._create_ssh_client(hostname, port, username, password, key_filepath)
        if not client:
            return False

        try:
            with client.open_scp() as scp:
                scp.get(remote_dir_str, local_path=local_dir_str, recursive=True)
            logger.info(f"Directory '{remote_dir_str}' downloaded successfully to '{local_dir_str}'")
            return True
        except paramiko.SCPException as e:
            logger.error(f"SCP error during directory download: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error during SCP directory download: {e}", exc_info=True)
            return False
        finally:
            if client:
                client.close()
                logger.debug("SSH client closed after scp_download_directory.")
