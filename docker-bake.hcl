variable "REGISTRY" {
    default = "docker.io"
}

variable "REGISTRY_USER" {
    default = "ashleykza"
}

variable "APP" {
    default = "runpod-worker-inswapper"
}

variable "RELEASE" {
    default = "7.0.2"
}

variable "CU_VERSION" {
    default = "130"
}

variable "CUDA_VERSION" {
    default = "13.0.3"
}

variable "TORCH_VERSION" {
    default = "2.14.0"
}

target "default" {
    dockerfile = "Dockerfile"
    tags = ["${REGISTRY}/${REGISTRY_USER}/${APP}:${RELEASE}"]
    annotations = [
        "org.opencontainers.image.description=Runpod Serverless worker for face swapping using FaceFusion swapper models and insightface detection",
    ]
    args = {
        RELEASE = "${RELEASE}"
        CUDA_VERSION = "${CUDA_VERSION}"
        INDEX_URL = "https://download.pytorch.org/whl/cu${CU_VERSION}"
        TORCH_VERSION = "${TORCH_VERSION}+cu${CU_VERSION}"
    }
}
