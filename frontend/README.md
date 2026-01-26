# Frontend

This directory is a sveltekit application that serves as an interface for the Onward Journey tool

## Developing

Once you've created a project and installed dependencies with `npm install`, define the following environment variables in a `.env` file:

```bash
PUBLIC_SUPPORT_CHAT_URL="<The websocket endpoint for genesys web messaging in the given region>"
PUBLIC_DEPLOYMENT_KEY="<The Web Messaging Deployment key for the chat to connect to>"
```

Then, run the server

```sh
npm run dev

# or start the server and open the app in a new browser tab
npm run dev -- --open
```

## Testing

You can run the jest unit tests by running:
```sh
npm run test
```

## Building

To create a production version of your app:

```sh
npm run build
```

You can preview the production build with `npm run preview`.

> To deploy your app, you may need to install an [adapter](https://svelte.dev/docs/kit/adapters) for your target environment.
