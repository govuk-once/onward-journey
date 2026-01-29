## LAMBDA BUILD PIPELINE: RDS SEEDER
# Automates dependency installation using 'uv' and zipping for the Data Ingestion layer.

resource "null_resource" "install_seeder_deps" {
  triggers = {
    # Re-run if the script changes
    python_code = filemd5("${path.module}/rds_seeder.py")
    # Also trigger if the .tool-versions changes (ensuring python version parity)
    tools_config = filemd5("${path.module}/../.tool-versions")
  }

  provisioner "local-exec" {
    command = "rm -rf ${path.module}/dist_package && mkdir -p ${path.module}/dist_package && uv pip install pg8000 --target ${path.module}/dist_package"
  }
}

resource "local_file" "seeder_script_copy" {
  content    = file("${path.module}/rds_seeder.py")
  filename   = "${path.module}/dist_package/rds_seeder.py"
  depends_on = [null_resource.install_seeder_deps]
}

data "archive_file" "rds_seeder_zip" {
  type        = "zip"
  source_dir  = "${path.module}/dist_package"
  output_path = "${path.module}/rds_seeder_payload.zip"

  depends_on = [
    null_resource.install_seeder_deps,
    local_file.seeder_script_copy
  ]
}
