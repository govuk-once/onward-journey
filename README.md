# Onward Journey

## Summary
A prototype for connecting users to further support after a chat session can't produce a satisfactory answer

## Getting started

### Pre-requisites

This project uses [mise-en-place](https://mise.jdx.dev/getting-started.html) to manage runtime versions

After installing `mise`, you should run `mise activate` from the root of this repo or [set up your shell to automatically active mise on startup](https://mise.jdx.dev/getting-started.html#activate-mise)

### Configure an AWS profile

Currently, we set up the infrastructure in the Made Tech sandbox AWS account until we have our AWS account (in progress)

Add the [mt-playground profile from the Made Tech handbook](https://github.com/madetech/handbook/blob/main/guides/cloud/aws_sandbox.md#cli-usage) and set up an `sso-session` for it by adding the following to your `~/.aws/config`:

```
[sso-session mt-playground]
sso_start_url = https://madetech.awsapps.com/start
sso_region = eu-west-2
sso_registration_scopes = sso:account:access
```

Then, set the `AWS_PROFILE` environment variable to use this AWS profile for deploying the infrastructure in a .env file:

```shell
echo 'AWS_PROFILE = "mt-playground"' >> .env
```

## Deploying infrastructure

Before you deploy infrastructure, you will need to make sure you are authenticated with AWS:

``shell
aws sso login --profile mt-platground
```

### Bootstrap

There is an `infrastructure/bootstrap` directory for managing the remote state in S3

### Application

To deploy changes to the Onward Journey application and services, you can use the tofu code in `infrastructure/application` and then `tofu plan` and `tofu apply` it

You can switch workspaces to deploy an entirely different instance, for example to test changes in an isolated environment without affecting the default workspace:
```shell
# View existing and selected workspace
tofu workspace list
# Create a new workspace
tofu workspace create foo
```

To view what changes your tofu code will make:
```shell
tofu plan
```

If you are happy with these changes:
```shell
tofu apply
```

To destroy an environment:
```shell
tofu destroy
```
