from langchain_community.agent_toolkits import FileManagementToolkit

def get_file_tools():
    toolkit = FileManagementToolkit(
        root_dir=str("."), # Restrict to current directory for safety, or make configurable
        selected_tools=["read_file", "write_file", "list_directory", "copy_file", "file_delete", "move_file"]
    )
    return toolkit.get_tools()
