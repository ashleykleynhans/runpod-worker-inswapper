## Building the Worker

You can either use my pre-built Docker image:
```
ashleykza/runpod-worker-inswapper:5.2.0
```

Or alternatively, you can build it yourself by following the
instructions below.

### Clone the repo

```bash
git clone https://github.com/ashleykleynhans/runpod-worker-inswapper.git
```

### Build the Docker image

```bash
cd runpod-worker-inswapper
docker build -t dockerhub-username/runpod-worker-inswapper:1.0.0 .
docker login
docker push dockerhub-username/runpod-worker-inswapper:1.0.0
```
