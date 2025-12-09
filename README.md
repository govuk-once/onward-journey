# Onward Journey

## Summary
A prototype for connecting users to further support after a chat session can't produce a satisfactory answer

## Getting started

### Pre-requisites

This project uses [mise-en-place](https://mise.jdx.dev/getting-started.html) to manage runtime versions

After installing `mise`, you should run `mise activate` from the root of this repo or [set up your shell to automatically activate mise on startup](https://mise.jdx.dev/getting-started.html#activate-mise)

You should also install all the tools from the [laptop-configuration repo](https://github.com/govuk-once/laptop-configuration)

Install the pre-commit hooks with `pre-commit install`. This will run the hooks listed in `.pre-commit-config.yaml` before each commit

## Running the application

There are instructions for running the prototype application in [the `app` directory README](app/README.md)


## Deploying infrastructure

You need to have the gds cli installed and configured to be able to deploy infrastructure, to the point that `gds aws once-onwardjourney-development-readonly -- echo "test"` succeeds

We use the gds cli to assume roles on our development machines, for a list of relevant roles see `gds aws | grep onwardjourney`. You use one of these roles when working with terraform by running e.g. `gds aws <role-name> -- terraform plan`, or you can run `gds aws <role-name> -- $SHELL` to start a new shell session authenticated as the relevant role

You will need to create a configuration file for the terraform backend for each target AWS account. Do this by copying `environments/.example.config` to `environments/<environment name>.config`, and filling in the placeholders

You will also need to set the environment name in `infrastructure/local.auto.tfvars`:
```shell
echo 'environment = "<environment name>" >> local.auto.tfvars'
```

You need to initialise the terraform in the `infrastructure/` directory before you can deploy. You will also need to run this when you change what AWS account you are targeting, e.g. from development to staging
```shell
# Run within an authenticated session from from `gds aws`
terraform init -reconfigure -backend-config=environments/<environment name>.config
```

You can switch workspaces to deploy an entirely different instance, for example to test changes in an isolated environment without affecting the default workspace:
```shell
# View existing and selected workspace
terraform workspace list
# Create a new workspace
terraform workspace create foo
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
