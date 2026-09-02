/*
 * long_run.c
 *
 * Simple long-running C program for demonstrating transparent
 * checkpoint/restart with DMTCP.
 *
 * IMPORTANT: This program contains NO application-level checkpoint code.
 * DMTCP will checkpoint and restore the process externally.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>


typedef struct {
    uint64_t iterations;
    uint64_t sleep_ms;
} options_t;


static void usage(const char *program)
{
    printf(
        "Usage: %s [OPTIONS]\n\n"
        "Options:\n"
        "  -n, --iterations N   Number of iterations (default: 120)\n"
        "      --sleep-ms N     Delay per iteration in ms (default: 1000)\n"
        "  -h, --help           Show this help\n",
        program
    );
}


static uint64_t parse_uint64(const char *text, const char *name)
{
    char *end = NULL;
    errno = 0;

    unsigned long long value = strtoull(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "Invalid value for %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }

    return (uint64_t) value;
}


static options_t parse_args(int argc, char **argv)
{
    options_t options = {
        .iterations = UINT64_C(120),
        .sleep_ms = UINT64_C(1000)
    };

    enum {
        OPT_SLEEP_MS = 1000
    };

    static const struct option long_options[] = {
        {"iterations", required_argument, NULL, 'n'},
        {"sleep-ms", required_argument, NULL, OPT_SLEEP_MS},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    int option;

    while ((option = getopt_long(
                argc,
                argv,
                "n:h",
                long_options,
                NULL
            )) != -1) {

        switch (option) {
            case 'n':
                options.iterations =
                    parse_uint64(optarg, "--iterations");
                break;

            case OPT_SLEEP_MS:
                options.sleep_ms =
                    parse_uint64(optarg, "--sleep-ms");
                break;

            case 'h':
                usage(argv[0]);
                exit(EXIT_SUCCESS);

            default:
                usage(argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (options.iterations == 0) {
        fprintf(stderr, "--iterations must be greater than zero\n");
        exit(EXIT_FAILURE);
    }

    return options;
}


static void sleep_milliseconds(uint64_t milliseconds)
{
    struct timespec request;

    request.tv_sec = (time_t) (milliseconds / 1000);
    request.tv_nsec =
        (long) ((milliseconds % 1000) * UINT64_C(1000000));

    while (nanosleep(&request, &request) != 0) {
        if (errno != EINTR) {
            perror("nanosleep");
            exit(EXIT_FAILURE);
        }
    }
}


int main(int argc, char **argv)
{
    options_t options = parse_args(argc, argv);

    char hostname[256] = "unknown";

    if (gethostname(hostname, sizeof(hostname)) != 0) {
        perror("gethostname");
        return EXIT_FAILURE;
    }

    hostname[sizeof(hostname) - 1] = '\0';

    uint64_t iteration = 0;
    double accumulator = 0.0;

    printf("============================================================\n");
    printf("DMTCP long-running demo application\n");
    printf("============================================================\n");
    printf("PID            : %ld\n", (long) getpid());
    printf("Host           : %s\n", hostname);
    printf("Iterations     : %" PRIu64 "\n", options.iterations);
    printf("Sleep/iter     : %" PRIu64 " ms\n", options.sleep_ms);
    printf("Checkpoint code: NONE\n");
    printf("============================================================\n");
    fflush(stdout);

    for (iteration = 1; iteration <= options.iterations; ++iteration) {
        /*
         * Keep a small piece of numerical state in memory in addition
         * to the loop counter. DMTCP will restore both transparently.
         */
        accumulator += sqrt((double) iteration);

        printf(
            "iteration = %4" PRIu64
            "   accumulator = %.8f\n",
            iteration,
            accumulator
        );
        fflush(stdout);

        sleep_milliseconds(options.sleep_ms);
    }

    printf("============================================================\n");
    printf("Calculation complete\n");
    printf("Final iteration   : %" PRIu64 "\n", options.iterations);
    printf("Final accumulator : %.8f\n", accumulator);
    printf("============================================================\n");

    return EXIT_SUCCESS;
}
