import asyncio
import threading
from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from vmlx_engine.image_requests import (ImageRequest, ImageRequestCancelled, cancel_image_request, current_image_request, image_request_scope)
from vmlx_engine.image_progress import observed_image_call

class Request:
    headers = {}
    disconnected = False
    async def is_disconnected(self):
        return self.disconnected

def test_idle_cancel_does_not_poison_next_request():
    assert cancel_image_request()["cancelled"] is False
    async def run():
        async with image_request_scope(Request(), {}, asyncio.Lock()) as state:
            state.check()
    asyncio.run(run())

def test_active_cancel_and_retry():
    async def run():
        lock = asyncio.Lock()
        with pytest.raises(HTTPException) as caught:
            async with image_request_scope(Request(), {"request_id":"active"}, lock) as state:
                assert cancel_image_request("wrong")["cancelled"] is False
                assert cancel_image_request("active")["cancelled"] is True
                state.check()
        assert caught.value.status_code == 409
        assert caught.value.detail["code"] == "image_generation_cancelled"
        assert not lock.locked()
        assert cancel_image_request("active")["cancelled"] is False
        async with image_request_scope(Request(), {"request_id":"retry"}, lock) as state:
            state.check()
    asyncio.run(run())

def test_queued_cancel_cannot_cancel_active_owner():
    async def run():
        lock=asyncio.Lock()
        async with image_request_scope(Request(), {"request_id":"first"}, lock) as first:
            async def queued():
                async with image_request_scope(Request(), {"request_id":"second"}, lock):
                    pytest.fail("cancelled queued request entered worker")
            task=asyncio.create_task(queued())
            await asyncio.sleep(0)
            assert cancel_image_request("second")["cancelled"] is True
            assert not first.cancelled.is_set()
            with pytest.raises(HTTPException) as caught: await asyncio.wait_for(task, .2)
            assert caught.value.detail["request_id"]=="second"
            assert lock.locked() and not first.cancelled.is_set()
        assert not lock.locked()
    asyncio.run(run())

def test_disconnect_cancels_only_own_request():
    async def run():
        request=Request()
        with pytest.raises(HTTPException):
            async with image_request_scope(request, {"request_id":"gone"}, asyncio.Lock()) as state:
                request.disconnected=True
                await asyncio.sleep(.12)
                state.check()
    asyncio.run(run())

def test_duplicate_active_id_is_not_replaced():
    async def run():
        lock=asyncio.Lock()
        async with image_request_scope(Request(), {"request_id":"same"}, lock) as first:
            with pytest.raises(HTTPException) as caught:
                async with image_request_scope(Request(), {"request_id":"same"}, lock): pass
            assert caught.value.detail["code"]=="image_request_id_in_use"
            assert not first.cancelled.is_set()
    asyncio.run(run())

def test_mflux_callback_cancels_at_checkpoint_and_cleans_up():
    class Registry:
        def __init__(self):
            self.before_loop=[];self.in_loop=[];self.after_loop=[];self.interrupt=[]
        def register(self,x):
            for name in ("before_loop","in_loop","after_loop","interrupt"):
                getattr(self,name).append(x)
    registry=Registry()
    existing=object();registry.in_loop.append(existing)
    state=ImageRequest("callback")
    calls=[]
    def generate_image(**kwargs):
        for t in range(5):
            calls.append(t)
            if t==1: state.cancelled.set()
            for cb in registry.in_loop:
                if hasattr(cb,"call_in_loop"): cb.call_in_loop(t=t)
        pytest.fail("cancelled diffusion completed")
    token=current_image_request.set(state)
    try:
        with pytest.raises(ImageRequestCancelled):
            observed_image_call(SimpleNamespace(callbacks=registry,generate_image=generate_image),model_name="fixture",model_class="Fixture",num_inference_steps=5)
    finally: current_image_request.reset(token)

    assert calls==[0,1]
    assert registry.in_loop==[existing]
    assert registry.before_loop==registry.after_loop==registry.interrupt==[]

@pytest.mark.parametrize("bad", [0, [], {}, "", " ", "bad\nline", "x"*129])
def test_request_ids_are_validated_before_registration(bad):
    async def run():
        with pytest.raises(HTTPException) as caught:
            async with image_request_scope(Request(), {"request_id":bad}, asyncio.Lock()): pass
        assert caught.value.status_code==400
    asyncio.run(run())

def test_production_executor_carries_request_context_and_drains_cancelled_work():
    import vmlx_engine.server as server
    original=server._image_gen_executor
    server._image_gen_executor=None
    entered=threading.Event(); release=threading.Event(); finished=threading.Event()
    def worker():
        state=current_image_request.get()
        assert state and state.request_id=="executor"
        entered.set()
        assert release.wait(5)
        finished.set()
    async def run():
        lock=asyncio.Lock()
        async def owner():
            async with image_request_scope(Request(), {"request_id":"executor"}, lock):
                await server._run_image_gen_call(worker)
        task=asyncio.create_task(owner())
        while not entered.is_set(): await asyncio.sleep(.01)
        task.cancel()
        await asyncio.sleep(.02)
        assert not task.done() and lock.locked() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError): await task
        assert finished.is_set() and not lock.locked()
    try: asyncio.run(run())
    finally:
        release.set()
        server._shutdown_image_gen_executor()
        server._image_gen_executor=original

def test_without_callbacks_drops_cancelled_result_after_return():
    state=ImageRequest("no-callbacks")
    def generate_image(**kwargs):
        state.cancelled.set()
        return object()
    token=current_image_request.set(state)
    try:
        with pytest.raises(ImageRequestCancelled):
            observed_image_call(SimpleNamespace(generate_image=generate_image),model_name="fixture",model_class="Fixture")
    finally: current_image_request.reset(token)
