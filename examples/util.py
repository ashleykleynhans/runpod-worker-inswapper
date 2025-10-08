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

    if resp_json['output'] is not None and 'image' in resp_json['output']:
        save_result_image(resp_json)
    else:
        print(json.dumps(resp_json, indent=4, default=str))

    print(f'Total time taken for RunPod Serverless API call {total_time} seconds')


def post_request(payload, runtype='runsync'):
    timer = Timer()
    env = dotenv_values('.env')
    runpod_api_key = env.get('RUNPOD_API_KEY', None)
    runpod_endpoint_id = env.get('RUNPOD_ENDPOINT_ID', None)

    if runpod_api_key is not None and runpod_endpoint_id is not None:
        base_url = f'https://api.runpod.ai/v2/{runpod_endpoint_id}'
    else:
        base_url = f'http://127.0.0.1:8000'

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
