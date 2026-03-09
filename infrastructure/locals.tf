locals {
  # Finds all files within the '../app/resources/oj_rag_data' folder and creates a list of file paths (e.g., 'oj_rag_data.csv') for Terraform to track.
  mock_data_files = fileset("../app/resources/oj_rag_data", "**")
}
