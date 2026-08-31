/*
 * pi_checkpoint.c
 *
 * Monte Carlo estimation of pi with application-level
 * checkpoint/restart.
 *
 * Demonstrates:
 *
 *   - explicit application state
 *   - binary checkpoint files
 *   - RNG-state preservation
 *   - temporary-file + atomic rename
 *   - manual restart using --resume
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>


#define CHECKPOINT_VERSION 1

static const double PI_TRUE =
    3.141592653589793238462643383279502884;


/*
 * ------------------------------------------------------------------
 * Checkpoint state
 * ------------------------------------------------------------------
 */

typedef struct {

    char magic[8];

    uint32_t version;

    uint64_t completed_darts;
    uint64_t inside_circle;

    uint64_t rng_state;

    uint64_t total_darts;
    uint64_t seed;

    double elapsed_seconds;

} checkpoint_t;


/*
 * Runtime options.
 */

typedef struct {

    uint64_t darts;
    uint64_t seed;

    uint64_t report_every;
    uint64_t checkpoint_every;

    double sleep_seconds;

    const char *checkpoint_file;

    int resume;

} options_t;


/*
 * ------------------------------------------------------------------
 * RNG
 * ------------------------------------------------------------------
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
    return (rng_next(state) >> 11) *
           (1.0 / 9007199254740992.0);
}


/*
 * ------------------------------------------------------------------
 * Timing
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
        (long) ((seconds - req.tv_sec) * 1.0e9);

    while (nanosleep(&req, &req) != 0) {

        if (errno != EINTR) {
            perror("nanosleep");
            exit(EXIT_FAILURE);
        }
    }
}


/*
 * ------------------------------------------------------------------
 * Checkpoint I/O
 * ------------------------------------------------------------------
 */

static void initialize_checkpoint(
    checkpoint_t *checkpoint
)
{
    memset(checkpoint, 0, sizeof(*checkpoint));

    memcpy(
        checkpoint->magic,
        "PICHECK",
        7
    );

    checkpoint->version =
        CHECKPOINT_VERSION;
}


static int checkpoint_is_valid(
    const checkpoint_t *checkpoint
)
{
    if (
        memcmp(
            checkpoint->magic,
            "PICHECK",
            7
        ) != 0
    ) {
        return 0;
    }

    if (
        checkpoint->version !=
        CHECKPOINT_VERSION
    ) {
        return 0;
    }

    return 1;
}


static void save_checkpoint(
    const char *filename,
    const checkpoint_t *checkpoint
)
{
    char tmp_filename[4096];

    int written = snprintf(
        tmp_filename,
        sizeof(tmp_filename),
        "%s.tmp",
        filename
    );

    if (
        written < 0 ||
        (size_t) written >= sizeof(tmp_filename)
    ) {
        fprintf(
            stderr,
            "Checkpoint filename is too long.\n"
        );
        exit(EXIT_FAILURE);
    }


    FILE *file =
        fopen(tmp_filename, "wb");

    if (file == NULL) {
        perror("fopen checkpoint");
        exit(EXIT_FAILURE);
    }


    if (
        fwrite(
            checkpoint,
            sizeof(*checkpoint),
            1,
            file
        ) != 1
    ) {

        perror("fwrite checkpoint");
        fclose(file);
        exit(EXIT_FAILURE);
    }


    /*
     * Flush C stdio buffers.
     */
    if (fflush(file) != 0) {
        perror("fflush checkpoint");
        fclose(file);
        exit(EXIT_FAILURE);
    }


    /*
     * Ask the OS to flush the underlying file.
     */
    if (fsync(fileno(file)) != 0) {
        perror("fsync checkpoint");
        fclose(file);
        exit(EXIT_FAILURE);
    }


    if (fclose(file) != 0) {
        perror("fclose checkpoint");
        exit(EXIT_FAILURE);
    }


    /*
     * Atomic replacement on the same filesystem.
     */
    if (
        rename(
            tmp_filename,
            filename
        ) != 0
    ) {
        perror("rename checkpoint");
        exit(EXIT_FAILURE);
    }
}


static checkpoint_t load_checkpoint(
    const char *filename
)
{
    checkpoint_t checkpoint;

    FILE *file =
        fopen(filename, "rb");

    if (file == NULL) {
        perror("fopen checkpoint");
        exit(EXIT_FAILURE);
    }


    if (
        fread(
            &checkpoint,
            sizeof(checkpoint),
            1,
            file
        ) != 1
    ) {

        fprintf(
            stderr,
            "Failed to read checkpoint file.\n"
        );

        fclose(file);
        exit(EXIT_FAILURE);
    }


    if (fclose(file) != 0) {
        perror("fclose checkpoint");
        exit(EXIT_FAILURE);
    }


    if (!checkpoint_is_valid(&checkpoint)) {

        fprintf(
            stderr,
            "Invalid or incompatible "
            "checkpoint file.\n"
        );

        exit(EXIT_FAILURE);
    }


    return checkpoint;
}


/*
 * ------------------------------------------------------------------
 * CLI helpers
 * ------------------------------------------------------------------
 */

static void usage(const char *program)
{
    printf(
        "Usage: %s [OPTIONS]\n\n"
        "Options:\n"
        "  -n, --darts N\n"
        "      --seed N\n"
        "      --report-every N\n"
        "      --checkpoint-every N\n"
        "      --checkpoint-file FILE\n"
        "      --sleep SEC\n"
        "      --resume\n"
        "  -h, --help\n",
        program
    );
}


static uint64_t parse_uint64(
    const char *text,
    const char *name
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
            name,
            text
        );

        exit(EXIT_FAILURE);
    }

    return (uint64_t) value;
}


static double parse_double(
    const char *text,
    const char *name
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
            name,
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
        .checkpoint_every = UINT64_C(100000),
        .sleep_seconds = 0.0,
        .checkpoint_file = "pi_checkpoint.bin",
        .resume = 0
    };


    enum {
        OPT_SEED = 1000,
        OPT_REPORT,
        OPT_CHECKPOINT_EVERY,
        OPT_CHECKPOINT_FILE,
        OPT_SLEEP,
        OPT_RESUME
    };


    static const struct option long_options[] = {

        {"darts", required_argument, NULL, 'n'},

        {"seed",
         required_argument,
         NULL,
         OPT_SEED},

        {"report-every",
         required_argument,
         NULL,
         OPT_REPORT},

        {"checkpoint-every",
         required_argument,
         NULL,
         OPT_CHECKPOINT_EVERY},

        {"checkpoint-file",
         required_argument,
         NULL,
         OPT_CHECKPOINT_FILE},

        {"sleep",
         required_argument,
         NULL,
         OPT_SLEEP},

        {"resume",
         no_argument,
         NULL,
         OPT_RESUME},

        {"help",
         no_argument,
         NULL,
         'h'},

        {NULL, 0, NULL, 0}
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

            case OPT_REPORT:
                options.report_every =
                    parse_uint64(
                        optarg,
                        "--report-every"
                    );
                break;

            case OPT_CHECKPOINT_EVERY:
                options.checkpoint_every =
                    parse_uint64(
                        optarg,
                        "--checkpoint-every"
                    );
                break;

            case OPT_CHECKPOINT_FILE:
                options.checkpoint_file =
                    optarg;
                break;

            case OPT_SLEEP:
                options.sleep_seconds =
                    parse_double(
                        optarg,
                        "--sleep"
                    );
                break;

            case OPT_RESUME:
                options.resume = 1;
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
            "--darts must be > 0\n"
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


    uint64_t completed_darts = 0;
    uint64_t inside_circle = 0;

    uint64_t rng_state =
        options.seed;

    double previous_elapsed = 0.0;


    /*
     * --------------------------------------------------------------
     * Restore checkpoint
     * --------------------------------------------------------------
     */

    if (options.resume) {

        checkpoint_t checkpoint =
            load_checkpoint(
                options.checkpoint_file
            );


        if (
            checkpoint.total_darts !=
            options.darts
        ) {

            fprintf(
                stderr,
                "Checkpoint belongs to a "
                "different calculation.\n"
            );

            exit(EXIT_FAILURE);
        }


        if (
            checkpoint.seed !=
            options.seed
        ) {

            fprintf(
                stderr,
                "Checkpoint uses a "
                "different seed.\n"
            );

            exit(EXIT_FAILURE);
        }


        completed_darts =
            checkpoint.completed_darts;

        inside_circle =
            checkpoint.inside_circle;

        rng_state =
            checkpoint.rng_state;

        previous_elapsed =
            checkpoint.elapsed_seconds;


        printf("\n");
        printf("========================================================================\n");
        printf("Restarting from checkpoint\n");
        printf("========================================================================\n");

        printf(
            "Completed darts  : %" PRIu64 "\n",
            completed_darts
        );

        printf(
            "Inside circle    : %" PRIu64 "\n",
            inside_circle
        );

        printf(
            "Previous runtime : %.2f s\n",
            previous_elapsed
        );

        printf("========================================================================\n\n");
    }


    const double start_time =
        wall_time();


    /*
     * --------------------------------------------------------------
     * Main loop
     * --------------------------------------------------------------
     */

    for (
        uint64_t i = completed_darts + 1;
        i <= options.darts;
        ++i
    ) {

        const double x =
            2.0 * rng_uniform(&rng_state)
            - 1.0;

        const double y =
            2.0 * rng_uniform(&rng_state)
            - 1.0;


        if (x * x + y * y <= 1.0) {
            ++inside_circle;
        }


        /*
         * Progress report
         */

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
                previous_elapsed +
                wall_time() -
                start_time;


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


        /*
         * ----------------------------------------------------------
         * Periodic checkpoint
         * ----------------------------------------------------------
         */

        if (
            options.checkpoint_every > 0 &&
            i % options.checkpoint_every == 0
        ) {

            checkpoint_t checkpoint;

            initialize_checkpoint(
                &checkpoint
            );


            checkpoint.completed_darts = i;

            checkpoint.inside_circle =
                inside_circle;

            checkpoint.rng_state =
                rng_state;

            checkpoint.total_darts =
                options.darts;

            checkpoint.seed =
                options.seed;

            checkpoint.elapsed_seconds =
                previous_elapsed +
                wall_time() -
                start_time;


            save_checkpoint(
                options.checkpoint_file,
                &checkpoint
            );


            printf(
                "  --> checkpoint saved "
                "at dart %" PRIu64 "\n",
                i
            );

            fflush(stdout);
        }
    }


    const double total_elapsed =
        previous_elapsed +
        wall_time() -
        start_time;


    const double pi_estimate =
        4.0 *
        (double) inside_circle /
        (double) options.darts;


    /*
     * Final checkpoint.
     */

    checkpoint_t final_checkpoint;

    initialize_checkpoint(
        &final_checkpoint
    );

    final_checkpoint.completed_darts =
        options.darts;

    final_checkpoint.inside_circle =
        inside_circle;

    final_checkpoint.rng_state =
        rng_state;

    final_checkpoint.total_darts =
        options.darts;

    final_checkpoint.seed =
        options.seed;

    final_checkpoint.elapsed_seconds =
        total_elapsed;


    save_checkpoint(
        options.checkpoint_file,
        &final_checkpoint
    );


    printf("\n");
    printf("========================================================================\n");
    printf("Final result\n");
    printf("========================================================================\n");

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
        total_elapsed
    );

    printf("========================================================================\n");

    return EXIT_SUCCESS;
}
