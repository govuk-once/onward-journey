locals {
  # Finds all files within the '../mock_data' folder and creates a list of file paths (e.g., 'mock_rag_data.csv') for Terraform to track.
  mock_data_files = fileset("../mock_data", "**")
}
