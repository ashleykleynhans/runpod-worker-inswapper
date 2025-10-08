# Runpod Testing

## Create and activate Python venv

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## Add Runpod credentials to .env

Copy the `.env.example` file to '.env' in the `examples`
directory as follows:

```bash
cd examples
cp .env.example .env
```

Edit the .env file and add your RunPod API key to
`RUNPOD_API_KEY` and your endpoint ID to
`RUNPOD_ENDPOINT_ID`.  Without these credentials,
the examples will attempt to run locally instead of
on RunPod.

## Run example scripts

Once the venv is created and activated, the requirements
installed, and the credentials added to the .env
file, you can run a script, for example:

```bash
python all_1_source_into_all_1_target.py
```

This will display the HTTP status code and the filename
of the output image, for example:
```
Status code: 200
Saving image: 792a7e9f-9c36-4d35-b408-0d45d8e2bbcb.jpg
Total time taken for RunPod Serverless API call 14.0 seconds
```

You can then open the output image (in this case
`792a7e9f-9c36-4d35-b408-0d45d8e2bbcb.jpg`) to view the
results of the face swap.

You obviously need to edit the payload within the
script to achieve the desired results.
