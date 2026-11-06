"""Download a declared paper plan into a private archive with immutable receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


def acquire(paper, output):
    dest = output / (paper['paper_id'] + '.pdf')
    receipt = output / (paper['paper_id'] + '.receipt.private.json')
    if dest.exists():
        saved = json.loads(receipt.read_text())
        if hashlib.sha256(dest.read_bytes()).hexdigest() != saved['sha256']:
            raise ValueError('archived_pdf_changed')
        return saved
    urls = [paper['download_url']] + paper.get('fallback_urls', [])
    if '/2022-08/' in urls[0]:
        urls.append(urls[0].replace('/2022-08/', '/2022-09/'))
    attempts = []
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (academic research)'})
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read(30_000_001)
                resolved, content_type = response.geturl(), response.headers.get('Content-Type')
            if len(data) > 30_000_000 or not data.startswith(b'%PDF-') or b'%%EOF' not in data[-4096:]:
                raise ValueError('incomplete_or_invalid_pdf')
            result = {'paper_id': paper['paper_id'], 'status': 'downloaded', 'pdf': str(dest),
                      'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                      'requested_url': url, 'resolved_url': resolved, 'content_type': content_type,
                      'retrieved_at': datetime.now(timezone.utc).isoformat(), 'earlier_attempts': attempts}
            dest.write_bytes(data)
            receipt.write_text(json.dumps(result, indent=2)+'\n')
            return result
        except Exception as error:
            attempts.append({'url': url, 'error_type': type(error).__name__, 'error': str(error)})
    return {'paper_id': paper['paper_id'], 'status': 'failed', 'attempts': attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError('use_new_attempt_receipt')
    plan = json.loads(args.plan.read_text())
    ids = [p['paper_id'] for p in plan['papers']]
    if len(ids) != len(set(ids)) or any(not i.replace('_', '').isalnum() for i in ids):
        raise ValueError('invalid_paper_ids')
    args.archive.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda p: acquire(p, args.archive), plan['papers']))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps({'protocol': plan['protocol'],
        'plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(), 'files': results}, indent=2)+'\n')
    for r in results:
        print(r['paper_id'], r['status'], r.get('bytes', ''))
    if any(r['status'] != 'downloaded' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
