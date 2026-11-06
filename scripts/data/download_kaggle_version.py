"""Read-only download of an explicitly selected Kaggle notebook output version.

Run with the Python environment containing Kaggle's SDK. Credentials remain in
Kaggle's external configuration; signed download URLs are never serialized.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def download(slug, version, output):
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest, ApiGetKernelRequest
    owner, notebook = slug.split('/')
    raw = output / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    files = {}; page_token = None; log_saved = False; seen_pages = set()
    with api.build_kaggle_client() as client:
        def current_metadata():
            req=ApiGetKernelRequest();req.user_name=owner;req.kernel_slug=notebook
            return client.kernels.kernels_api_client.get_kernel(req).metadata
        metadata=current_metadata()
        if metadata.current_version_number != version:
            raise ValueError('current_version_differs_no_latest_fallback_allowed')
        explicit=True
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = notebook
            if explicit: request.version_label = str(version)
            request.page_size = 200
            if page_token: request.page_token = page_token
            try:
                response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            except requests.HTTPError as error:
                if not explicit or error.response.status_code not in (400,404): raise
                # Some Kaggle API deployments reject numeric version labels.
                # A latest lookup is permitted only while SDK metadata matches
                # the requested version, checked both before and after download.
                explicit=False
                request.version_label=None
                response=client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                name = item.file_name
                destination = (raw / name).resolve()
                if not destination.is_relative_to(raw.resolve()):
                    raise ValueError('unsafe_output_path')
                reply = requests.get(item.url, timeout=180)
                reply.raise_for_status()
                body = reply.content
                if destination.exists() and destination.read_bytes() != body:
                    raise ValueError('existing_download_differs_from_explicit_version')
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(body)
                files[name] = hashlib.sha256(body).hexdigest()
            if response.log and not log_saved:
                body = response.log.encode()
                name = notebook + '.log'
                destination = raw / name
                if destination.exists() and destination.read_bytes() != body:
                    raise ValueError('existing_log_differs_from_explicit_version')
                destination.write_bytes(body)
                files[name] = hashlib.sha256(body).hexdigest()
                log_saved = True
            page_token = response.next_page_token
            if not page_token: break
            if page_token in seen_pages: raise ValueError('repeated_paging_token')
            seen_pages.add(page_token)
        if current_metadata().current_version_number != version:
            raise ValueError('version_changed_during_download_no_receipt_created')
    if not files or not log_saved: raise ValueError('missing_outputs_or_log')
    receipt = dict(slug=slug, version=version, version_explicitly_verified_via_SDK=True,
        version_request_field='version_label' if explicit else 'SDK current_version_number checked before and after',
        current_version_number=metadata.current_version_number,
        is_private=metadata.is_private,enable_gpu=metadata.enable_gpu,
        enable_internet=metadata.enable_internet,machine_shape=metadata.machine_shape,files=files,
        downloader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = output / 'download_manifest.json'
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError('refuse_overwrite_download_receipt')
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print('Downloaded explicit version %d: %d files including log. No GPU job submitted.' % (version, len(files)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--slug',required=True)
    p.add_argument('--version',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        if args.version < 1: raise ValueError('positive_version_required')
        download(args.slug,args.version,args.output)
    except Exception as error:
        # Provider errors can contain signed URLs; never print arbitrary messages.
        print('Download stopped. Error type: '+type(error).__name__,file=sys.stderr)
        sys.exit(1)


if __name__=='__main__':main()
