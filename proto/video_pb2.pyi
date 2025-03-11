from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Frame(_message.Message):
    __slots__ = ("image_data", "timestamp")
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    image_data: bytes
    timestamp: int
    def __init__(self, image_data: _Optional[bytes] = ..., timestamp: _Optional[int] = ...) -> None: ...

class UploadStatus(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class StreamRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
