import io
import time
import json
import uuid
import base64
import requests
from PIL import Image
from dotenv import dotenv_values

OUTPUT_FORMAT = 'JPEG'
STATUS_IN_QUEUE = 'IN_QUEUE'
STATUS_IN_PROGRESS = 'IN_PROGRESS'
STATUS_FAILED = 'FAILED'
STATUS_CANCELLED = 'CANCELLED'
STATUS_COMPLETED = 'COMPLETED'
STATUS_TIMED_OUT = 'TIMED_OUT'


class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_elapsed_time(self):
        end = time.time()
        return round(end - self.start, 1)


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return str(base64.b64encode(image_file.read()).decode('utf-8'))


def save_result_image(resp_json):
    img = Image.open(io.BytesIO(base64.b64decode(resp_json['output']['image'])))
    file_extension = 'jpeg' if OUTPUT_FORMAT == 'JPEG' else 'png'
    output_file = f'{uuid.uuid4()}.{file_extension}'

    with open(output_file, 'wb') as f:
        print(f'Saving image: {output_file}')
        img.save(f, format=OUTPUT_FORMAT)


def handle_response(resp_json, timer):
    total_time = timer.get_elapsed_time()

    output = resp_json.get('output', {})

    if not output:
        print("No output found in the response.")

    # Calculate response size
    response_json = json.dumps(resp_json)
    response_size_bytes = len(response_json.encode('utf-8'))
    _, response_size_str = calculate_payload_size(resp_json)
    print(f'Response size: {response_size_str} ({response_size_bytes:,} bytes)')

    if 'image' in output:
        # Calculate the size of just the base64 image
        image_size_bytes = len(output['image'])
        if image_size_bytes < 1024 * 1024:
            image_size_kb = image_size_bytes / 1024
            print(f'Image size (base64): {image_size_kb:.2f} KB ({image_size_bytes:,} bytes)')
        else:
            image_size_mb = image_size_bytes / (1024 * 1024)
            print(f'Image size (base64): {image_size_mb:.2f} MB ({image_size_bytes:,} bytes)')

        save_result_image(resp_json)
    else:
        print(json.dumps(resp_json, indent=4, default=str))

    print(f'Total time taken for RunPod Serverless API call {total_time} seconds')


def calculate_payload_size(payload):
    """Calculate the size of the payload in bytes and return human-readable format"""
    payload_json = json.dumps(payload)
    size_bytes = len(payload_json.encode('utf-8'))

    # Convert to human-readable format
    if size_bytes < 1024:
        return size_bytes, f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_kb = size_bytes / 1024
        return size_bytes, f"{size_kb:.2f} KB"
    else:
        size_mb = size_bytes / (1024 * 1024)
        return size_bytes, f"{size_mb:.2f} MB"


def post_request(payload, runtype='runsync'):
    timer = Timer()
    env = dotenv_values('.env')
    runpod_api_key = env.get('RUNPOD_API_KEY', None)
    runpod_endpoint_id = env.get('RUNPOD_ENDPOINT_ID', None)

    if runpod_api_key is not None and runpod_endpoint_id is not None:
        base_url = f'https://api.runpod.ai/v2/{runpod_endpoint_id}'
    else:
        base_url = f'http://127.0.0.1:8000'

    # Calculate and display payload size
    size_bytes, size_str = calculate_payload_size(payload)
    print(f'Payload size: {size_str} ({size_bytes:,} bytes)')

    r = requests.post(
        f'{base_url}/{runtype}',
        headers={
            'Authorization': f'Bearer {runpod_api_key}'
        },
        json=payload
    )

    print(f'Status code: {r.status_code}')

    if r.status_code == 200:
        resp_json = r.json()

        if 'output' in resp_json:
            handle_response(resp_json, timer)
        else:
            job_status = resp_json['status']
            print(f'Job status: {job_status}')

            if job_status == STATUS_IN_QUEUE or job_status == STATUS_IN_PROGRESS:
                request_id = resp_json['id']
                request_in_queue = True

                while request_in_queue:
                    r = requests.get(
                        f'{base_url}/status/{request_id}',
                        headers={
                            'Authorization': f'Bearer {runpod_api_key}'
                        },
                    )

                    print(f'Status code from RunPod status endpoint: {r.status_code}')

                    if r.status_code == 200:
                        resp_json = r.json()
                        job_status = resp_json['status']

                        if job_status == STATUS_IN_QUEUE or job_status == STATUS_IN_PROGRESS:
                            print(f'RunPod request {request_id} is {job_status}, sleeping for 5 seconds...')
                            time.sleep(5)
                        elif job_status == STATUS_FAILED:
                            request_in_queue = False
                            print(f'RunPod request {request_id} failed')
                            print(json.dumps(resp_json, indent=4, default=str))
                        elif job_status == STATUS_COMPLETED:
                            request_in_queue = False
                            print(f'RunPod request {request_id} completed')
                            handle_response(resp_json, timer)
                        elif job_status == STATUS_TIMED_OUT:
                            request_in_queue = False
                            print(f'ERROR: RunPod request {request_id} timed out')
                        else:
                            request_in_queue = False
                            print(f'ERROR: Invalid status response from RunPod status endpoint')
                            print(json.dumps(resp_json, indent=4, default=str))
            elif job_status == STATUS_COMPLETED \
                    and 'output' in resp_json \
                    and 'status' in resp_json['output'] \
                    and resp_json['output']['status'] == 'error':
                print(f'ERROR: {resp_json["output"]["message"]}')
            else:
                print(json.dumps(resp_json, indent=4, default=str))
    else:
        print(f'ERROR: {r.content}')
