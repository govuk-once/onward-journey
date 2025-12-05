# Onward Journey

## Summary
A prototype for connecting users to further support after a chat session can't produce a satisfactory answer

## Getting started

### Pre-requisites

This project uses [mise-en-place](https://mise.jdx.dev/getting-started.html) to manage runtime versions

After installing `mise`, you should run `mise activate` from the root of this repo or [set up your shell to automatically active mise on startup](https://mise.jdx.dev/getting-started.html#activate-mise)

### Configure an AWS profile

Set up a `profile` and `sso-session` for the AWS account you will be setting up the infrastructure with. See the [aws cli documentation](https://awscli.amazonaws.com/v2/documentation/api/2.8.7/reference/configure/sso.html) for reference

Then, set the `AWS_PROFILE` environment variable to use this AWS profile for deploying the infrastructure in a .env file:

```shell
echo 'AWS_PROFILE = "<profile-name>"' >> .env
```

## Deploying infrastructure

Before you deploy infrastructure, you will need to make sure you are authenticated with AWS:

```shell
aws sso login --profile <profile-name>
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
