from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CameraFrame(_message.Message):
    __slots__ = ("camera_id", "frame")
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    camera_id: str
    frame: Frame
    def __init__(self, camera_id: _Optional[str] = ..., frame: _Optional[_Union[Frame, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("camera_id",)
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    camera_id: str
    def __init__(self, camera_id: _Optional[str] = ...) -> None: ...

class ListCamerasRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListCamerasResponse(_message.Message):
    __slots__ = ("camera_ids",)
    CAMERA_IDS_FIELD_NUMBER: _ClassVar[int]
    camera_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, camera_ids: _Optional[_Iterable[str]] = ...) -> None: ...
