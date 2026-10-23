"""Read only selected members of a public ZIP using validated HTTP byte ranges."""
import io
import re

import requests


class RangeReader(io.RawIOBase):
    def __init__(self, url, session=None):
        self.url = url
        self.session = session or requests.Session()
        self.position = 0
        self.bytes_downloaded = 0
        self.tail, start, end, self.size = self._fetch("bytes=-65536")
        self.tail_start = start

    def _fetch(self, spec):
        # Distinct query prevents caches from reusing a response for a different range.
        with self.session.get(self.url, params={"range_key": spec}, headers={"Range": spec},
                              timeout=45, stream=True) as response:
            if response.status_code != 206:
                raise ValueError(f"Range request returned {response.status_code}; refusing full download")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match:
                raise ValueError("Missing/invalid Content-Range")
            start, end, size = map(int, match.groups())
            if not (0 <= start <= end < size) or end - start + 1 > 16 * 1024 * 1024:
                raise ValueError("Invalid or oversized range")
            a, b = spec.removeprefix("bytes=").split("-")
            expected = (max(0, size-int(b)), size-1) if not a else (int(a), min(int(b), size-1))
            if (start, end) != expected:
                raise ValueError("Server returned wrong byte range")
            chunks, count = [], 0
            for chunk in response.iter_content(65536):
                count += len(chunk)
                if count > end - start + 1:
                    raise ValueError("Range body exceeds declared length")
                chunks.append(chunk)
            data = b"".join(chunks)
            if len(data) != end - start + 1:
                raise ValueError("Truncated range body")
            self.bytes_downloaded += len(data)
            return data, start, end, size

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.position+n)
        if end <= self.position:
            return b""
        if end-self.position > 16*1024*1024:
            raise ValueError("Read exceeds 16 MB safety bound")
        if self.position >= self.tail_start:
            data = self.tail[self.position-self.tail_start:end-self.tail_start]
        else:
            data, _, _, size = self._fetch(f"bytes={self.position}-{end-1}")
            if size != self.size:
                raise ValueError("Remote archive size changed")
        self.position = end
        return data


def read_member(archive, name):
    info = archive.getinfo(name)
    if info.file_size > 12*1024*1024 or info.compress_size > 12*1024*1024:
        raise ValueError("Member exceeds image-size bound")
    return archive.read(name)  # zipfile also checks the member CRC.
