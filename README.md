# My Voice

Clone a voice from a 10-30 second recording and make it say anything. Runs on Railway,
installs as an app from Chrome.

- Open the Railway link, sign in (user `me`, password = `APP_PASSWORD`).
- Chrome: the install icon in the address bar (phone: menu -> *Add to Home screen* / *Install app*).
- Record or upload a voice, save it with a name, type text, press **Speak**, download the result.

## Railway setup
- Built from this repo's `Dockerfile` (CPU; no graphics card on Railway, so a sentence takes several seconds).
- Volume mounted at `/data`: the model (~3 GB, downloaded on first start) and saved voices live there.
- Variables: `APP_PASSWORD` (required, the app will not start without it), optional `APP_USERNAME` (default `me`).
- First start takes a few minutes while the model downloads; later starts are quick.

Model: [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox), MIT licence. It adds an inaudible watermark.
