# Docker Guide for Multi-Object Tracking API

This guide explains how to build and run the Multi-Object Tracking API using Docker.

## Prerequisites

Ensure you have the following installed on your system:
- [Docker](https://docs.docker.com/get-docker/)


## Building the Docker Image

1. Clone the repository if you haven’t already:
   ```sh
   git clone https://github.com/your-repo/multi-object-tracking.git
   cd multi-object-tracking
   ```

2. Build the Docker image:
   ```sh
   docker build -t multi-object-tracking .
   ```

This will:
- Install all necessary dependencies.
- Include `ffmpeg` for audio and video processing.
- Expose the correct ports for FastAPI.

## Running the Container

To run the container:
```sh
   docker run -p 8000:8000 -v $(pwd)/static:/app/static --name tracking-container multi-object-tracking
```

### Explanation:
- `-p 8000:8000` maps the container’s port `8000` to your local machine.
- `-v $(pwd)/static:/app/static` mounts the `static` directory to persist videos.
- `--name tracking-container` assigns a name to the container.

## Using Docker Compose (Optional)
If you prefer **Docker Compose**, create a `docker-compose.yml` file and run:
```sh
docker-compose up --build
```

## Accessing the API
Once the container is running, access the API at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Upload page**: [http://localhost:8000](http://localhost:8000)

## Stopping and Removing the Container
To stop the container:
```sh
docker stop tracking-container
```
To remove the container:
```sh
docker rm tracking-container
```

## Troubleshooting
- If the container doesn’t start, check logs:
  ```sh
  docker logs tracking-container
  ```
- Ensure Docker has the necessary permissions to bind volumes.

---
Now you're ready to use the Multi-Object Tracking API in Docker! 🚀

