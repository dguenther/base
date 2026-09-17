//! This module contains a blocking runtime for futures, allowing for synchronous execution of async
//! code in an embedded environment.

use core::future::Future;
#[cfg(feature = "std")]
use core::{
    pin::pin,
    task::{Context, Poll, Waker},
};
#[cfg(feature = "std")]
extern crate std;
#[cfg(feature = "std")]
use std::sync::LazyLock;

/// Runtime used when `block_on` is called outside of a Tokio runtime.
#[cfg(feature = "std")]
static FALLBACK_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build fallback runtime")
});

/// This function blocks on a future in place until it is ready.
///
/// When called from within a Tokio runtime, the future is first polled once with a no-op waker.
/// Futures that complete immediately (e.g. in-memory oracle lookups) return without handing the
/// worker off via `block_in_place`; only futures that are still pending pay that cost.
#[cfg(feature = "std")]
pub fn block_on<T>(f: impl Future<Output = T>) -> T {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        // Fallback to a shared runtime if we're not in one
        return FALLBACK_RUNTIME.block_on(f);
    };

    let mut f = pin!(f);
    if let Poll::Ready(v) = f.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        return v;
    }

    tokio::task::block_in_place(|| runtime.block_on(f))
}

/// This function busy waits on a future until it is ready. It uses a no-op waker to poll the future
/// in a thread-blocking loop.
#[cfg(not(feature = "std"))]
pub fn block_on<T>(f: impl Future<Output = T>) -> T {
    use alloc::boxed::Box;
    use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    let mut f = Box::pin(f);

    // Construct a no-op waker.
    fn noop_clone(_: *const ()) -> RawWaker {
        noop_raw_waker()
    }
    const fn noop(_: *const ()) {}
    fn noop_raw_waker() -> RawWaker {
        let vtable = &RawWakerVTable::new(noop_clone, noop, noop, noop);
        RawWaker::new(core::ptr::null(), vtable)
    }
    // SAFETY: The waker is a no-op waker that does nothing. It is safe to construct from the
    // raw waker as the vtable functions are valid no-ops.
    let waker = unsafe { Waker::from_raw(noop_raw_waker()) };
    let mut context = Context::from_waker(&waker);

    loop {
        // Safety: This is safe because we only poll the future once per loop iteration,
        // and we do not move the future after pinning it.
        if let Poll::Ready(v) = f.as_mut().poll(&mut context) {
            return v;
        }
    }
}

#[cfg(test)]
mod tests {
    use core::future::ready;

    use super::*;

    #[test]
    fn test_block_on() {
        let f = async { 42 };
        assert_eq!(block_on(f), 42);
    }

    #[test]
    fn test_block_on_ready() {
        let f = ready(42);
        assert_eq!(block_on(f), 42);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_block_on_pending_outside_runtime() {
        let f = async {
            tokio::time::sleep(core::time::Duration::from_millis(1)).await;
            42
        };
        assert_eq!(block_on(f), 42);
    }

    #[cfg(feature = "std")]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_block_on_pending_inside_runtime() {
        let f = async {
            tokio::time::sleep(core::time::Duration::from_millis(1)).await;
            42
        };
        assert_eq!(block_on(f), 42);
    }
}
