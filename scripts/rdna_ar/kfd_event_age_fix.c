#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <linux/kfd_ioctl.h>

#define KFD_MAX_SIGNAL_EVENTS 65536

// Global lock-free age cache indexed by event_id.
// Initialized to 0. 64-bit aligned for lock-free atomic ops on x86_64.
static uint64_t g_event_age_cache[KFD_MAX_SIGNAL_EVENTS] __attribute__((aligned(8))) = {0};

static int (*g_real_ioctl)(int fd, unsigned long request, void *arg) = NULL;

__attribute__((constructor))
static void init_kfd_fix(void) {
    g_real_ioctl = (int (*)(int, unsigned long, void *))dlsym(RTLD_NEXT, "ioctl");
}

int ioctl(int fd, unsigned long request, void *arg) {
    if (__builtin_expect(!g_real_ioctl, 0)) {
        g_real_ioctl = (int (*)(int, unsigned long, void *))dlsym(RTLD_NEXT, "ioctl");
    }

    if (__builtin_expect(request == AMDKFD_IOC_WAIT_EVENTS, 0)) {
        struct kfd_ioctl_wait_events_args *args = (struct kfd_ioctl_wait_events_args *)arg;
        if (args && args->events_ptr && args->num_events > 0) {
            struct kfd_event_data *evts = (struct kfd_event_data *)(uintptr_t)args->events_ptr;

            // Before ioctl: if userspace supplied a stale in_age (e.g. 1) due to the ROCR stack
            // reset bug, replace it with the most recent observed age from previous completions.
            for (uint32_t i = 0; i < args->num_events; i++) {
                uint32_t id = evts[i].event_id;
                if (id < KFD_MAX_SIGNAL_EVENTS) {
                    uint64_t cached = __atomic_load_n(&g_event_age_cache[id], __ATOMIC_RELAXED);
                    if (cached > evts[i].signal_event_data.last_event_age) {
                        evts[i].signal_event_data.last_event_age = cached;
                    }
                }
            }

            int ret = g_real_ioctl(fd, request, arg);

            // After ioctl: update the age cache with the latest event age returned by the KFD driver.
            for (uint32_t i = 0; i < args->num_events; i++) {
                uint32_t id = evts[i].event_id;
                if (id < KFD_MAX_SIGNAL_EVENTS) {
                    uint64_t out_age = evts[i].signal_event_data.last_event_age;
                    uint64_t cur = __atomic_load_n(&g_event_age_cache[id], __ATOMIC_RELAXED);
                    while (out_age > cur) {
                        if (__atomic_compare_exchange_n(&g_event_age_cache[id], &cur, out_age,
                                                        false, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
                            break;
                        }
                    }
                }
            }
            return ret;
        }
    }

    return g_real_ioctl(fd, request, arg);
}
