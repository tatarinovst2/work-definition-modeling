from enum import Enum


class ResponseCode(Enum):
    SUCCESS = 200
    BAD_REQUEST = 400
    NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502


STATUSES = {
    ResponseCode.SUCCESS.value: {
        "status": "success",
        "code": ResponseCode.SUCCESS.value
    },
    ResponseCode.BAD_REQUEST.value: {
        "status": "failure",
        "reason": "Bad Request",
        "code": ResponseCode.BAD_REQUEST.value
    },
    ResponseCode.NOT_FOUND.value: {
        "status": "failure",
        "reason": "Not Found",
        "code": ResponseCode.NOT_FOUND.value
    },
    ResponseCode.INTERNAL_SERVER_ERROR.value: {
        "status": "failure",
        "reason": "Internal Server Error",
        "code": ResponseCode.INTERNAL_SERVER_ERROR.value
    },
    ResponseCode.BAD_GATEWAY.value: {
        "status": "failure",
        "reason": "Bad Gateway",
        "code": ResponseCode.BAD_GATEWAY.value
    },
}
