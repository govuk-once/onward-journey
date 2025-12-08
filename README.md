# Onward Journey

## Summary
A prototype for connecting users to further support after a chat session can't produce a satisfactory answer

## Getting started

### Pre-requisites

This project uses [mise-en-place](https://mise.jdx.dev/getting-started.html) to manage runtime versions

After installing `mise`, you should run `mise activate` from the root of this repo or [set up your shell to automatically active mise on startup](https://mise.jdx.dev/getting-started.html#activate-mise)

You should also install all the tools from the [laptop-configuration repo](https://github.com/govuk-once/laptop-configuration)

## Deploying infrastructure

You need to have the gds cli installed and configured to be able to deploy infrastructure, to the point that `gds aws once-onwardjourney-development-readonly -- echo "test"` succeeds

We use the gds cli to assume roles on our development machines, for a list of relevant roles see `gds aws | grep onwardjourney`. You use one of these roles when working with terraform by running e.g. `gds aws <role-name> -- terraform plan`, or you can run `gds aws <role-name> -- $SHELL` to start a new shell session authenticated as the relevant role.

### Bootstrap

There is an `infrastructure/bootstrap` directory for managing the remote state in S3

### Application

To deploy changes to the Onward Journey application and services, you can use the terraform code in `infrastructure/application` and then `terraform plan` and `terraform apply` it

You can switch workspaces to deploy an entirely different instance, for example to test changes in an isolated environment without affecting the default workspace:
```shell
# View existing and selected workspace
terraform workspace list
# Create a new workspace
terraform workspace create foo
```

To initialise the terraform directory before deploying:
```shell
terraform init
```

To view what changes your terraform code will make:
```shell
terraform plan
```

If you are happy with these changes:
```shell
terraform apply
```

To destroy an environment:
```shell
terraform destroy
```
