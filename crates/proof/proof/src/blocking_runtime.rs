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

    /// Ready only once woken by the waker from the *latest* poll. Hangs if `block_on` keeps
    /// waiting on the waker from the initial no-op poll.
    #[cfg(feature = "std")]
    struct WakeLater {
        polls: u32,
        done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    #[cfg(feature = "std")]
    impl Future for WakeLater {
        type Output = u32;

        fn poll(mut self: core::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
            if self.done.load(std::sync::atomic::Ordering::SeqCst) {
                return Poll::Ready(self.polls);
            }
            self.polls += 1;
            let waker = cx.waker().clone();
            let done = std::sync::Arc::clone(&self.done);
            std::thread::spawn(move || {
                std::thread::sleep(core::time::Duration::from_millis(5));
                done.store(true, std::sync::atomic::Ordering::SeqCst);
                waker.wake();
            });
            Poll::Pending
        }
    }

    #[cfg(feature = "std")]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_block_on_reregisters_waker_after_noop_poll() {
        let f = WakeLater { polls: 0, done: Default::default() };
        // First poll uses the no-op waker, second registers the real one.
        assert_eq!(block_on(f), 2);
    }

    #[cfg(feature = "std")]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_block_on_oneshot_sent_between_polls() {
        for _ in 0..1000 {
            let (tx, rx) = tokio::sync::oneshot::channel();
            std::thread::spawn(move || tx.send(7).unwrap());
            assert_eq!(block_on(rx).unwrap(), 7);
        }
    }

    #[cfg(feature = "std")]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_block_on_yield_now_inside_runtime() {
        assert_eq!(
            block_on(async {
                tokio::task::yield_now().await;
                42
            }),
            42
        );
    }

    #[cfg(feature = "std")]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_block_on_nested_inside_runtime() {
        let v = block_on(async {
            let inner = block_on(async {
                tokio::time::sleep(core::time::Duration::from_millis(1)).await;
                block_on(ready(1))
            });
            inner + block_on(async { 41 })
        });
        assert_eq!(v, 42);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_block_on_nested_outside_runtime() {
        let v = block_on(async {
            tokio::time::sleep(core::time::Duration::from_millis(1)).await;
            block_on(async {
                tokio::task::yield_now().await;
                42
            })
        });
        assert_eq!(v, 42);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_block_on_ready_inside_current_thread_runtime() {
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async { assert_eq!(block_on(ready(42)), 42) });
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_block_on_from_spawn_blocking() {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
        let v = rt.block_on(async {
            tokio::task::spawn_blocking(|| {
                block_on(async {
                    tokio::time::sleep(core::time::Duration::from_millis(1)).await;
                    42
                })
            })
            .await
        });
        assert_eq!(v.unwrap(), 42);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_block_on_concurrent_fallback_runtime() {
        let handles: alloc::vec::Vec<_> = (0..32)
            .map(|i| {
                std::thread::spawn(move || {
                    let mut sum = 0u64;
                    for j in 0..100u64 {
                        sum += block_on(async move {
                            tokio::task::yield_now().await;
                            i * j
                        });
                    }
                    sum
                })
            })
            .collect();
        for (i, h) in handles.into_iter().enumerate() {
            assert_eq!(h.join().unwrap(), (i as u64) * 4950);
        }
    }

    #[cfg(feature = "std")]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_block_on_panic_in_first_poll_propagates() {
        let r = std::panic::catch_unwind(|| block_on(async { panic!("boom") }));
        assert!(r.is_err());
        // Runtime is still usable afterwards.
        assert_eq!(block_on(ready(1)), 1);
    }

    #[cfg(feature = "std")]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_block_on_drops_future_once() {
        struct DropCounter(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let drops = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let guard = DropCounter(std::sync::Arc::clone(&drops));
        block_on(async move {
            let _g = guard;
            tokio::time::sleep(core::time::Duration::from_millis(1)).await;
        });
        assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
