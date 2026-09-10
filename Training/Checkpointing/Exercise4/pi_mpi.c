#define _POSIX_C_SOURCE 200809L

#include <mpi.h>

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const double PI_TRUE =
    3.141592653589793238462643383279502884;

/*
 * xorshift64* pseudo-random number generator.
 *
 * Each MPI rank gets its own independent RNG state.
 */
static uint64_t xorshift64star(uint64_t *state)
{
    uint64_t x = *state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;

    *state = x;

    return x * UINT64_C(2685821657736338717);
}


/*
 * Convert a 64-bit random integer to a double in [0,1).
 */
static double random_uniform(uint64_t *state)
{
    return (xorshift64star(state) >> 11) *
           (1.0 / 9007199254740992.0);
}


/*
 * Sleep for the requested number of milliseconds.
 *
 * This is included only to make the computation slow enough
 * for checkpoint/restart demonstrations.
 */
static void sleep_ms(long milliseconds)
{
    if (milliseconds <= 0) {
        return;
    }

    struct timespec req;

    req.tv_sec = milliseconds / 1000;
    req.tv_nsec = (milliseconds % 1000) * 1000000L;

    while (nanosleep(&req, &req) == -1 && errno == EINTR) {
        /* continue sleeping for remaining time */
    }
}


static void usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Options:\n"
            "  --darts N          Total number of Monte Carlo darts\n"
            "                     Default: 100000000\n"
            "\n"
            "  --seed N           Base random-number seed\n"
            "                     Default: 12345\n"
            "\n"
            "  --report-every N   Report progress every N local darts\n"
            "                     Default: 1000000\n"
            "\n"
            "  --sleep-ms N       Sleep N milliseconds after each report\n"
            "                     Default: 0\n"
            "\n"
            "  --help             Show this help message\n",
            program);
}


static uint64_t parse_uint64(const char *text, const char *option)
{
    char *end = NULL;

    errno = 0;

    unsigned long long value = strtoull(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr,
                "ERROR: invalid value for %s: %s\n",
                option,
                text);
        exit(EXIT_FAILURE);
    }

    return (uint64_t)value;
}


static long parse_long(const char *text, const char *option)
{
    char *end = NULL;

    errno = 0;

    long value = strtol(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr,
                "ERROR: invalid value for %s: %s\n",
                option,
                text);
        exit(EXIT_FAILURE);
    }

    return value;
}


int main(int argc, char **argv)
{
    uint64_t total_darts = UINT64_C(100000000);
    uint64_t base_seed = UINT64_C(12345);
    uint64_t report_every = UINT64_C(1000000);

    long report_sleep_ms = 0;

    /*
     * Parse command-line options before MPI_Init().
     *
     * Every MPI rank receives the same command line.
     */
    for (int i = 1; i < argc; ++i) {

        if (strcmp(argv[i], "--darts") == 0) {

            if (++i >= argc) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }

            total_darts =
                parse_uint64(argv[i], "--darts");

        } else if (strcmp(argv[i], "--seed") == 0) {

            if (++i >= argc) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }

            base_seed =
                parse_uint64(argv[i], "--seed");

        } else if (strcmp(argv[i], "--report-every") == 0) {

            if (++i >= argc) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }

            report_every =
                parse_uint64(argv[i], "--report-every");

        } else if (strcmp(argv[i], "--sleep-ms") == 0) {

            if (++i >= argc) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }

            report_sleep_ms =
                parse_long(argv[i], "--sleep-ms");

        } else if (strcmp(argv[i], "--help") == 0) {

            usage(argv[0]);
            return EXIT_SUCCESS;

        } else {

            fprintf(stderr,
                    "ERROR: unknown option: %s\n",
                    argv[i]);

            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }


    if (total_darts == 0) {
        fprintf(stderr,
                "ERROR: --darts must be greater than zero\n");
        return EXIT_FAILURE;
    }

    if (report_every == 0) {
        fprintf(stderr,
                "ERROR: --report-every must be greater than zero\n");
        return EXIT_FAILURE;
    }

    if (report_sleep_ms < 0) {
        fprintf(stderr,
                "ERROR: --sleep-ms cannot be negative\n");
        return EXIT_FAILURE;
    }


    MPI_Init(&argc, &argv);


    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    /*
     * Divide the total number of darts among MPI ranks.
     *
     * The first 'remainder' ranks receive one extra dart.
     */
    uint64_t base_darts =
        total_darts / (uint64_t)size;

    uint64_t remainder =
        total_darts % (uint64_t)size;

    uint64_t local_darts =
        base_darts +
        ((uint64_t)rank < remainder ? 1 : 0);


    /*
     * Maximum local workload.
     *
     * All ranks remain in the reporting loop until the rank
     * with the largest workload finishes. This guarantees
     * that MPI_Reduce is called the same number of times by
     * every rank.
     */
    uint64_t max_local_darts =
        base_darts + (remainder > 0 ? 1 : 0);


    /*
     * Give each MPI rank a deterministic but different seed.
     */
    uint64_t rng_state =
        base_seed +
        UINT64_C(0x9e3779b97f4a7c15) *
        (uint64_t)(rank + 1);

    if (rng_state == 0) {
        rng_state = UINT64_C(1);
    }


    uint64_t local_completed = 0;
    uint64_t local_inside = 0;

    double start_time = MPI_Wtime();


    if (rank == 0) {

        printf("============================================================\n");
        printf("MPI Monte Carlo Pi\n");
        printf("============================================================\n");
        printf("MPI ranks          : %d\n", size);
        printf("Total darts        : %" PRIu64 "\n", total_darts);
        printf("Base seed          : %" PRIu64 "\n", base_seed);
        printf("Report every       : %" PRIu64 " local darts\n",
               report_every);
        printf("Sleep after report : %ld ms\n", report_sleep_ms);
        printf("============================================================\n");

        fflush(stdout);
    }


    /*
     * Work in blocks so that every rank reaches the MPI_Reduce
     * calls at the same logical points.
     */
    for (uint64_t block_start = 0;
         block_start < max_local_darts;
         block_start += report_every) {

        uint64_t block_end =
            block_start + report_every;

        if (block_end > local_darts) {
            block_end = local_darts;
        }


        /*
         * Ranks that have already finished simply skip the
         * computation but still participate in MPI_Reduce.
         */
        if (block_start < local_darts) {

            for (uint64_t i = block_start;
                 i < block_end;
                 ++i) {

                double x =
                    2.0 * random_uniform(&rng_state) - 1.0;

                double y =
                    2.0 * random_uniform(&rng_state) - 1.0;

                if (x * x + y * y <= 1.0) {
                    ++local_inside;
                }

                ++local_completed;
            }
        }


        uint64_t global_completed = 0;
        uint64_t global_inside = 0;


        MPI_Reduce(&local_completed,
                   &global_completed,
                   1,
                   MPI_UINT64_T,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);

        MPI_Reduce(&local_inside,
                   &global_inside,
                   1,
                   MPI_UINT64_T,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);


        if (rank == 0 && global_completed > 0) {

            double pi_estimate =
                4.0 *
                (double)global_inside /
                (double)global_completed;

            double error =
                pi_estimate - PI_TRUE;

            double elapsed =
                MPI_Wtime() - start_time;

            double percent =
                100.0 *
                (double)global_completed /
                (double)total_darts;

            printf("Progress: %6.2f%%"
                   "  darts=%" PRIu64
                   "  pi=%.10f"
                   "  error=%+.3e"
                   "  elapsed=%.2f s\n",
                   percent,
                   global_completed,
                   pi_estimate,
                   error,
                   elapsed);

            fflush(stdout);
        }


        /*
         * Deliberately slow the program for workshop demos.
         */
        sleep_ms(report_sleep_ms);
    }


    uint64_t global_inside = 0;
    uint64_t global_completed = 0;


    MPI_Reduce(&local_inside,
               &global_inside,
               1,
               MPI_UINT64_T,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);

    MPI_Reduce(&local_completed,
               &global_completed,
               1,
               MPI_UINT64_T,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);


    if (rank == 0) {

        double elapsed =
            MPI_Wtime() - start_time;

        double pi_estimate =
            4.0 *
            (double)global_inside /
            (double)global_completed;

        double error =
            pi_estimate - PI_TRUE;

        printf("\n");
        printf("============================================================\n");
        printf("Finished\n");
        printf("============================================================\n");
        printf("Darts       : %" PRIu64 "\n", global_completed);
        printf("Inside      : %" PRIu64 "\n", global_inside);
        printf("Pi estimate : %.12f\n", pi_estimate);
        printf("Pi true     : %.12f\n", PI_TRUE);
        printf("Error       : %+.6e\n", error);
        printf("Elapsed     : %.3f s\n", elapsed);
        printf("============================================================\n");

        fflush(stdout);
    }


    MPI_Finalize();

    return EXIT_SUCCESS;
}
