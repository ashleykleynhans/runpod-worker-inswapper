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
    default = "6.0.14"
}

variable "CU_VERSION" {
    default = "124"
}

variable "CUDA_VERSION" {
    default = "12.4.1"
}

variable "TORCH_VERSION" {
    default = "2.6.0"
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
