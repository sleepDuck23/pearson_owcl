"""Read selected members of dataset ZIPs without transferring the full archive."""

import io
import re
from urllib.request import Request, urlopen


class HTTPRangeReader(io.RawIOBase):
    """Seekable byte stream for servers that support HTTP range requests."""

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.position = 0
        self.etag = None
        _, self.size, self.etag = self._read_range(0, 1)

    def _read_range(self, start, length):
        end = start + length - 1
        headers = {'Range': f'bytes={start}-{end}'}
        if self.etag is not None:
            headers['If-Range'] = self.etag
        with urlopen(Request(self.url, headers=headers), timeout=60) as response:
            content_range = response.headers.get('Content-Range', '')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
            if response.status != 206 or match is None:
                raise OSError('The server did not return an HTTP byte range')
            actual_start, actual_end, size = map(int, match.groups())
            if (actual_start, actual_end) != (start, end) or size <= end:
                raise OSError(f'Unexpected HTTP byte range {content_range}')
            data = response.read(length + 1)
            if len(data) != length:
                raise OSError(f'Incomplete HTTP byte range {content_range}')
            return data, size, response.headers.get('ETag')

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        if whence not in origin or origin[whence] + offset < 0:
            raise ValueError('Invalid seek position')
        self.position = origin[whence] + offset
        return self.position

    def readinto(self, buffer):
        length = min(len(buffer), self.size - self.position)
        if length <= 0:
            return 0
        data, size, etag = self._read_range(self.position, length)
        if size != self.size or etag != self.etag:
            raise OSError('The remote ZIP changed during download')
        buffer[:length] = data
        self.position += length
        return length
