# Image request cancellation

Supply a unique printable `request_id` (up to 128 characters) in an image
generation/edit JSON or multipart request. `X-Image-Request-ID` is accepted as
a fallback when the body omits it. Concurrent duplicate IDs return 409.

Send `POST /v1/images/cancel` with `{"request_id":"your-id"}` and the same
server authentication. This targets only that registered request, including
a queued request. Unknown IDs return `cancelled: false`. Legacy empty-body
cancellation targets the currently active request only; idle cancellation
does not affect a later request.

The acknowledgement's `state: cancelling` is not worker completion. The
original endpoint returns 409 with `detail.code: image_generation_cancelled`
after the worker reaches a cooperative boundary. A running Metal operation
cannot be preempted. Adapters with mflux callbacks check cancellation before,
during and after denoising; adapters without callbacks can only discard the
result after their call returns. Queued cancellations never change another
request's cancellation state. Client disconnects cancel only their owner.

The app keeps the submission busy and displays Cancelling until the original
request finishes. Cancelled work is not appended to image history. Request IDs
are also logged alongside the internal image-job ID for correlation.
