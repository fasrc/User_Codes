/*
 * pi_naive.c
 *
 * Monte Carlo estimation of pi.
 *
 * Baseline version:
 *     - command-line arguments
 *     - deterministic PRNG
 *     - progress reporting
 *     - NO checkpoint/restart
 *
 * Example:
 *
 *   ./pi_naive \
 *       --darts 50000000 \
 *       --seed 42 \
 *       --report-every 1000000 \
 *       --sleep 1
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


static const double PI_TRUE =
    3.141592653589793238462643383279502884;


typedef struct {
    uint64_t darts;
    uint64_t seed;
    uint64_t report_every;
    double sleep_seconds;
} options_t;


/*
 * ------------------------------------------------------------------
 * Random-number generator
 * ------------------------------------------------------------------
 *
 * We use xorshift64* instead of rand().
 *
 * The entire RNG state is one uint64_t. This becomes useful in the
 * checkpointed version because restoring the stochastic calculation
 * requires restoring this value.
 */

static uint64_t rng_next(uint64_t *state)
{
    uint64_t x = *state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;

    *state = x;

    return x * UINT64_C(2685821657736338717);
}


static double rng_uniform(uint64_t *state)
{
    /*
     * Generate a double in [0, 1).
     *
     * IEEE double precision has 53 bits of significand precision,
     * so use the upper 53 bits of the random integer.
     */
    return (rng_next(state) >> 11) *
           (1.0 / 9007199254740992.0);
}


/*
 * ------------------------------------------------------------------
 * Timing helpers
 * ------------------------------------------------------------------
 */

static double wall_time(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    return (double) ts.tv_sec +
           (double) ts.tv_nsec * 1.0e-9;
}


static void sleep_seconds(double seconds)
{
    if (seconds <= 0.0) {
        return;
    }

    struct timespec req;

    req.tv_sec = (time_t) seconds;

    req.tv_nsec =
        (long) ((seconds - (double) req.tv_sec) * 1.0e9);

    while (nanosleep(&req, &req) != 0) {

        if (errno != EINTR) {
            perror("nanosleep");
            exit(EXIT_FAILURE);
        }
    }
}


/*
 * ------------------------------------------------------------------
 * Command-line parsing
 * ------------------------------------------------------------------
 */

static void usage(const char *program)
{
    printf(
        "Usage: %s [OPTIONS]\n\n"
        "Monte Carlo estimation of pi.\n\n"
        "Options:\n"
        "  -n, --darts N          Number of darts\n"
        "      --seed N           RNG seed (must be nonzero)\n"
        "      --report-every N   Report progress every N darts\n"
        "      --sleep SEC        Delay after each progress report\n"
        "  -h, --help             Show this help\n\n",
        program
    );
}


static uint64_t parse_uint64(
    const char *text,
    const char *option_name
)
{
    char *end = NULL;

    errno = 0;

    unsigned long long value =
        strtoull(text, &end, 10);

    if (
        errno != 0 ||
        end == text ||
        *end != '\0'
    ) {
        fprintf(
            stderr,
            "Invalid value for %s: %s\n",
            option_name,
            text
        );

        exit(EXIT_FAILURE);
    }

    return (uint64_t) value;
}


static double parse_double(
    const char *text,
    const char *option_name
)
{
    char *end = NULL;

    errno = 0;

    double value =
        strtod(text, &end);

    if (
        errno != 0 ||
        end == text ||
        *end != '\0'
    ) {
        fprintf(
            stderr,
            "Invalid value for %s: %s\n",
            option_name,
            text
        );

        exit(EXIT_FAILURE);
    }

    return value;
}


static options_t parse_args(
    int argc,
    char **argv
)
{
    options_t options = {
        .darts = UINT64_C(1000000),
        .seed = UINT64_C(42),
        .report_every = UINT64_C(100000),
        .sleep_seconds = 0.0
    };

    enum {
        OPT_SEED = 1000,
        OPT_REPORT_EVERY,
        OPT_SLEEP
    };

    static const struct option long_options[] = {

        {
            "darts",
            required_argument,
            NULL,
            'n'
        },

        {
            "seed",
            required_argument,
            NULL,
            OPT_SEED
        },

        {
            "report-every",
            required_argument,
            NULL,
            OPT_REPORT_EVERY
        },

        {
            "sleep",
            required_argument,
            NULL,
            OPT_SLEEP
        },

        {
            "help",
            no_argument,
            NULL,
            'h'
        },

        {
            NULL,
            0,
            NULL,
            0
        }
    };

    int option;

    while (
        (option = getopt_long(
            argc,
            argv,
            "n:h",
            long_options,
            NULL
        )) != -1
    ) {

        switch (option) {

            case 'n':

                options.darts =
                    parse_uint64(
                        optarg,
                        "--darts"
                    );

                break;


            case OPT_SEED:

                options.seed =
                    parse_uint64(
                        optarg,
                        "--seed"
                    );

                break;


            case OPT_REPORT_EVERY:

                options.report_every =
                    parse_uint64(
                        optarg,
                        "--report-every"
                    );

                break;


            case OPT_SLEEP:

                options.sleep_seconds =
                    parse_double(
                        optarg,
                        "--sleep"
                    );

                break;


            case 'h':

                usage(argv[0]);

                exit(EXIT_SUCCESS);


            default:

                usage(argv[0]);

                exit(EXIT_FAILURE);
        }
    }


    if (options.darts == 0) {

        fprintf(
            stderr,
            "--darts must be greater than zero\n"
        );

        exit(EXIT_FAILURE);
    }


    if (options.seed == 0) {

        fprintf(
            stderr,
            "--seed must be nonzero\n"
        );

        exit(EXIT_FAILURE);
    }


    if (options.sleep_seconds < 0.0) {

        fprintf(
            stderr,
            "--sleep cannot be negative\n"
        );

        exit(EXIT_FAILURE);
    }


    return options;
}


/*
 * ------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------
 */

int main(int argc, char **argv)
{
    options_t options =
        parse_args(argc, argv);

    uint64_t rng_state =
        options.seed;

    uint64_t inside_circle = 0;

    const double start_time =
        wall_time();


    printf(
        "========================================================================\n"
    );

    printf(
        "Monte Carlo Pi Estimation - C\n"
    );

    printf(
        "========================================================================\n"
    );


    printf(
        "Number of darts : %" PRIu64 "\n",
        options.darts
    );

    printf(
        "Random seed     : %" PRIu64 "\n",
        options.seed
    );

    printf(
        "Report interval : %" PRIu64 "\n",
        options.report_every
    );

    printf(
        "Sleep interval  : %.3f s\n",
        options.sleep_seconds
    );


    printf(
        "========================================================================\n"
    );


    for (
        uint64_t i = 1;
        i <= options.darts;
        ++i
    ) {

        const double x =
            2.0 * rng_uniform(&rng_state) - 1.0;

        const double y =
            2.0 * rng_uniform(&rng_state) - 1.0;


        if (x * x + y * y <= 1.0) {

            ++inside_circle;
        }


        if (
            options.report_every > 0 &&
            (
                i % options.report_every == 0 ||
                i == options.darts
            )
        ) {

            const double pi_estimate =
                4.0 *
                (double) inside_circle /
                (double) i;


            const double elapsed =
                wall_time() - start_time;


            printf(
                "darts = %12" PRIu64
                " / %" PRIu64
                "   pi = %.8f"
                "   error = %.3e"
                "   elapsed = %.2f s\n",
                i,
                options.darts,
                pi_estimate,
                fabs(pi_estimate - PI_TRUE),
                elapsed
            );


            fflush(stdout);


            sleep_seconds(
                options.sleep_seconds
            );
        }
    }


    const double elapsed =
        wall_time() - start_time;


    const double pi_estimate =
        4.0 *
        (double) inside_circle /
        (double) options.darts;


    printf("\n");

    printf(
        "========================================================================\n"
    );

    printf(
        "Final result\n"
    );

    printf(
        "========================================================================\n"
    );


    printf(
        "Darts thrown       : %" PRIu64 "\n",
        options.darts
    );

    printf(
        "Inside circle      : %" PRIu64 "\n",
        inside_circle
    );

    printf(
        "Estimated pi       : %.10f\n",
        pi_estimate
    );

    printf(
        "Reference pi       : %.10f\n",
        PI_TRUE
    );

    printf(
        "Absolute error     : %.6e\n",
        fabs(pi_estimate - PI_TRUE)
    );

    printf(
        "Total elapsed time : %.2f s\n",
        elapsed
    );


    printf(
        "========================================================================\n"
    );


    return EXIT_SUCCESS;
}
