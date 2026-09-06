CC ?= gcc
CFLAGS ?= -O3 -fPIC -shared -Wall -Wextra

all: vendor/kfd_event_age_fix.so

vendor/kfd_event_age_fix.so: kfd_event_age_fix.c
	mkdir -p vendor
	$(CC) $(CFLAGS) $< -o $@ -ldl

clean:
	rm -f vendor/kfd_event_age_fix.so

.PHONY: all clean
