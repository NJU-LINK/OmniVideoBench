CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

# Image arguments
IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN_PATCH = -300
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Video arguments
VIDEO_TOKEN_INDEX = -201
DEFAULT_VIDEO_TOKEN = "<video>"
NUM_FRAMES = 8
MAX_FRAMES = 32
NUM_FRAMES_PER_SECOND = 1

# Audio arguments
AUDIO_TOKEN_INDEX = -202
DEFAULT_AUDIO_TOKEN = "<audio>"

MODAL_INDEX_MAP = {
    "<audio>": -202,
    "<video>": -201,
    "<image>": -200,
}

MODAL_INDEX_REMAP = {v: k for k, v in MODAL_INDEX_MAP.items()}
DEFAULT_X_START_TOKEN = {'IMAGE': "<im_start>", 'VIDEO': "<vi_start>", 'AUDIO': "<au_start>", 'THERMAL': "<th_start>", 'DEPTH': "<de_start>"}
DEFAULT_X_END_TOKEN = {'IMAGE': "<im_end>", 'VIDEO': "<vi_end>", 'AUDIO': "<au_end>", 'THERMAL': "<th_end>", 'DEPTH': "<de_end>"}