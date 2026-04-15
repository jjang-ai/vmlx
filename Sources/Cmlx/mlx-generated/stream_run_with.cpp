#include "mlx/c/stream.h"
#include "mlx/c/device.h"

extern "C" int mlx_stream_run_with(
    mlx_stream stream,
    void (*callback)(void* context),
    void* context) {

    mlx_device dev = mlx_device_new();
    mlx_get_default_device(&dev);

    mlx_stream old_stream = mlx_stream_new();
    int rc = mlx_get_default_stream(&old_stream, dev);
    if (rc != 0) {
        mlx_stream_free(old_stream);
        mlx_device_free(dev);
        return rc;
    }

    rc = mlx_set_default_stream(stream);
    if (rc != 0) {
        mlx_stream_free(old_stream);
        mlx_device_free(dev);
        return rc;
    }

    callback(context);

    mlx_set_default_stream(old_stream);
    mlx_stream_free(old_stream);
    mlx_device_free(dev);
    return 0;
}
