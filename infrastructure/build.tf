## LAMBDA BUILD PIPELINE: RDS SEEDER
# Automates dependency installation using 'uv' and zipping for the Data Ingestion layer.

resource "null_resource" "install_seeder_deps" {
  triggers = {
    # Re-run if the script changes
    python_code = filemd5("${path.module}/../app/rds_seeder.py")
    # Also trigger if the .tool-versions changes (ensuring python version parity)
    tools_config = filemd5("${path.module}/../.tool-versions")
  }

  provisioner "local-exec" {
    command = <<EOT
      rm -rf ${path.module}/../dist/seeder_staging
      mkdir -p ${path.module}/../dist/seeder_staging
      uv pip install pg8000 --target ${path.module}/../dist/seeder_staging
    EOT
  }
}

resource "local_file" "seeder_script_copy" {
  content    = file("${path.module}/../app/rds_seeder.py")
  filename   = "${path.module}/../dist/seeder_staging/rds_seeder.py"
  depends_on = [null_resource.install_seeder_deps]
}

data "archive_file" "rds_seeder_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../dist/seeder_staging"
  output_path = "${path.module}/../dist/rds_seeder_payload.zip"

  depends_on = [
    null_resource.install_seeder_deps,
    local_file.seeder_script_copy
  ]
}


# TODO: If new version of Orchestrator Lambda build works, remove old version that re-installed every package after a small code change

# ## LAMBDA BUILD: ORCHESTRATOR
# # Prepares a Linux-compatible deployment package by bundling code and binary dependencies into a single ZIP.

# # Prunes uv.lock to get only Orchestrator-specific deps and packages them.
# resource "null_resource" "install_orchestrator_deps" {
#   triggers = {
#     # Re-run if the python logic, dependency lockfile, or this build script changes
#     python_code  = filemd5("${path.module}/../app/orchestrator.py")
#     lock_file    = filemd5("${path.module}/../app/uv.lock")
#     build_script = sha1(local.orchestrator_build_command)
#   }

#   provisioner "local-exec" {
#     command = local.orchestrator_build_command
#   }
# }

# locals {
#   orchestrator_build_command = <<EOT
#     # 1. Clean and enter app directory
#     rm -rf ${path.module}/../dist/orchestrator_staging
#     mkdir -p ${path.module}/../dist/orchestrator_staging
#     cd ${path.module}/../app

#     # 2. COMPILE for the exact Lambda architecture
#     # Use --python-platform for the resolver
#     uv pip compile pyproject.toml \
#       --python-platform x86_64-manylinux_2_28 \
#       --python-version 3.12 \
#       --output-file requirements_lambda.txt

#     # 3. INSTALL with strict platform enforcement
#     # FIX: Changed --platform to --python-platform
#     # Added --link-mode copy to ensure we get real files, not Mac symlinks
#     uv pip install \
#       --target ../dist/orchestrator_staging \
#       --python-platform x86_64-manylinux_2_28 \
#       --python-version 3.12 \
#       --only-binary=:all: \
#       --link-mode copy \
#       --no-cache \
#       -r requirements_lambda.txt

#     # 4. Copy logic and cleanup
#     cp orchestrator.py ../dist/orchestrator_staging/
#     [ -f requirements_lambda.txt ] && rm requirements_lambda.txt
#   EOT
# }


# data "archive_file" "orchestrator_zip" {
#   type        = "zip"
#   source_dir  = "${path.module}/../dist/orchestrator_staging"
#   output_path = "${path.module}/../dist/orchestrator_payload.zip"

#   depends_on = [
#     null_resource.install_orchestrator_deps
#   ]
# }

## LAMBDA BUILD: ORCHESTRATOR
# Prepares a Linux-compatible deployment package by bundling code and binary dependencies.

# 1. DEPENDENCY LAYER: Only runs when libraries or the build logic change.
resource "null_resource" "install_orchestrator_deps" {
  triggers = {
    # Re-run ONLY if dependencies change or the install logic is modified
    lock_file    = filemd5("${path.module}/../app/uv.lock")
    build_script = sha1(local.orchestrator_dep_install_command)
    # SAFETY: Check for the orchestrator script.
    # If it's missing, the folder was likely deleted, so force a rebuild.
    dir_exists = fileexists("${path.module}/../dist/orchestrator_staging/orchestrator.py") ? "exists" : timestamp()
  }

  provisioner "local-exec" {
    command = local.orchestrator_dep_install_command
  }
}

# 2. SCRIPT LAYER: Swaps the orchestrator.py logic (Very fast).
resource "null_resource" "sync_orchestrator_script" {
  triggers = {
    # Re-run every time the python code is modified
    python_code = filemd5("${path.module}/../app/orchestrator.py")
  }

  # Ensure the staging directory and dependencies exist first
  depends_on = [null_resource.install_orchestrator_deps]

  provisioner "local-exec" {
    # We do NOT use 'rm -rf' here. We keep the site-packages and just overwrite the script.
    command = "cp ${path.module}/../app/orchestrator.py ${path.module}/../dist/orchestrator_staging/"
  }
}

locals {
  orchestrator_dep_install_command = <<EOT
    # 1. Clean and prepare fresh staging directory for dependencies
    rm -rf ${path.module}/../dist/orchestrator_staging
    mkdir -p ${path.module}/../dist/orchestrator_staging
    cd ${path.module}/../app

    # 2. COMPILE for the exact Lambda architecture (Amazon Linux 2023)
    uv pip compile pyproject.toml \
      --python-platform x86_64-manylinux_2_28 \
      --python-version 3.12 \
      --output-file requirements_lambda.txt

    # 3. INSTALL with strict platform enforcement and no-cache
    uv pip install \
      --target ../dist/orchestrator_staging \
      --python-platform x86_64-manylinux_2_28 \
      --python-version 3.12 \
      --only-binary=:all: \
      --link-mode copy \
      --no-cache \
      -r requirements_lambda.txt

    # 4. Cleanup temporary requirements file
    [ -f requirements_lambda.txt ] && rm requirements_lambda.txt
  EOT
}

data "archive_file" "orchestrator_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../dist/orchestrator_staging"
  output_path = "${path.module}/../dist/orchestrator_payload.zip"

  # This is the "Anchor": It prevents the data source from finishing
  # until the uv install and script sync are 100% done.
  depends_on = [
    null_resource.install_orchestrator_deps,
    null_resource.sync_orchestrator_script
  ]
}

## LAMBDA BUILD: RDS TOOL (MCP SERVER)
# Bundles the database search tool with binary dependencies for the Gateway.

resource "null_resource" "install_rds_tool_deps" {
  triggers = {
    # Re-run if the tool logic or dependency lockfile changes
    python_code = filemd5("${path.module}/../app/rds_tool.py")
    lock_file   = filemd5("${path.module}/../app/uv.lock")
  }

  provisioner "local-exec" {
    # 1. Clean staging | 2. Install Linux binaries | 3. Copy logic
    command = <<EOT
      rm -rf ${path.module}/../dist/rds_tool_staging
      mkdir -p ${path.module}/../dist/rds_tool_staging
      cd ${path.module}/../app
      uv pip install --target ../dist/rds_tool_staging --platform manylinux_2_28_x86_64 --only-binary=:all: -- psycopg2-binary
      cp rds_tool.py ../dist/rds_tool_staging/
    EOT
  }
}

data "archive_file" "rds_tool_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../dist/rds_tool_staging"
  output_path = "${path.module}/../dist/rds_tool_payload.zip"

  depends_on = [
    null_resource.install_rds_tool_deps
  ]
}
