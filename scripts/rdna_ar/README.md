
## ROCm Event-Age Interposer (`vendor/kfd_event_age_fix.so`)
Linux KFD ABI 1.14+ introduces monotonic `event_age` checks in `AMDKFD_IOC_WAIT_EVENTS`. Upstream `ROCR-Runtime` resets `event_age = 1` on the stack, which causes `init_event_waiter` to immediately return `KFD_IOC_WAIT_RESULT_COMPLETE` in 3µs once an event has been signaled. This produces a ~3.4M ioctl/s busy-wait loop pinning 100% CPU on each GPU rank at idle.

`kfd_event_age_fix.so` intercepts `AMDKFD_IOC_WAIT_EVENTS`, caches the true kernel event ages, and supplies them to allow the thread to enter true sleep (`schedule_timeout`). This drops idle CPU load on the background GPU event threads to 0.0%.

Build with:
```bash
make -C scripts/rdna_ar
```
Auto-loaded by `scripts/rdna-serve.sh` via `LD_PRELOAD`.
