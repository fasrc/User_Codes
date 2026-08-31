/*
 * pi_signal.c
 *
 * Monte Carlo pi calculation with:
 *
 *   - periodic checkpointing
 *   - manual --resume
 *   - SIGUSR1 handling
 *   - SIGTERM handling
 *
 * Signal handlers only set flags. Checkpoint I/O occurs from
 * normal program execution.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>


#define CHECKPOINT_VERSION 1

static const double PI_TRUE =
    3.141592653589793238462643383279502884;


static volatile sig_atomic_t stop_requested = 0;
static volatile sig_atomic_t received_signal = 0;


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


typedef struct {
    uint64_t darts;
    uint64_t seed;
    uint64_t report_every;
    uint64_t checkpoint_every;

    double sleep_seconds;

    const char *checkpoint_file;
    int resume;
} options_t;


/* ------------------------------------------------------------------ */
/* Signal handling                                                     */
/* ------------------------------------------------------------------ */

static void signal_handler(int signum)
{
    stop_requested = 1;
    received_signal = signum;
}


static void install_signal_handlers(void)
{
    struct sigaction action;

    memset(&action, 0, sizeof(action));

    action.sa_handler = signal_handler;

    if (sigemptyset(&action.sa_mask) != 0) {
        perror("sigemptyset");
        exit(EXIT_FAILURE);
    }

    /*
     * Do not use SA_RESTART here. We want blocking calls such as
     * nanosleep() to return when the signal arrives.
     */
    action.sa_flags = 0;

    if (sigaction(SIGUSR1, &action, NULL) != 0) {
        perror("sigaction SIGUSR1");
        exit(EXIT_FAILURE);
    }

    if (sigaction(SIGTERM, &action, NULL) != 0) {
        perror("sigaction SIGTERM");
        exit(EXIT_FAILURE);
    }
}


static const char *signal_name(int signum)
{
    if (signum == SIGUSR1) {
        return "SIGUSR1";
    }

    if (signum == SIGTERM) {
        return "SIGTERM";
    }

    return "UNKNOWN";
}


/* ------------------------------------------------------------------ */
/* RNG                                                                 */
/* ------------------------------------------------------------------ */

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


/* ------------------------------------------------------------------ */
/* Timing                                                              */
/* ------------------------------------------------------------------ */

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

        if (errno == EINTR) {

            if (stop_requested) {
                return;
            }

            continue;
        }

        perror("nanosleep");
        exit(EXIT_FAILURE);
    }
}


/* ------------------------------------------------------------------ */
/* Checkpoint I/O                                                      */
/* ------------------------------------------------------------------ */

static void initialize_checkpoint(checkpoint_t *checkpoint)
{
    memset(checkpoint, 0, sizeof(*checkpoint));

    memcpy(checkpoint->magic, "PICHECK", 7);

    checkpoint->version = CHECKPOINT_VERSION;
}


static int checkpoint_is_valid(const checkpoint_t *checkpoint)
{
    return (
        memcmp(checkpoint->magic, "PICHECK", 7) == 0 &&
        checkpoint->version == CHECKPOINT_VERSION
    );
}


static void save_checkpoint(
    const char *filename,
    const checkpoint_t *checkpoint
)
{
    char tmp_filename[4096];

    int n = snprintf(
        tmp_filename,
        sizeof(tmp_filename),
        "%s.tmp",
        filename
    );

    if (n < 0 || (size_t) n >= sizeof(tmp_filename)) {
        fprintf(stderr, "Checkpoint filename is too long.\n");
        exit(EXIT_FAILURE);
    }

    FILE *fp = fopen(tmp_filename, "wb");

    if (fp == NULL) {
        perror("fopen checkpoint");
        exit(EXIT_FAILURE);
    }

    if (fwrite(
            checkpoint,
            sizeof(*checkpoint),
            1,
            fp
        ) != 1) {

        perror("fwrite checkpoint");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (fflush(fp) != 0) {
        perror("fflush checkpoint");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (fsync(fileno(fp)) != 0) {
        perror("fsync checkpoint");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (fclose(fp) != 0) {
        perror("fclose checkpoint");
        exit(EXIT_FAILURE);
    }

    if (rename(tmp_filename, filename) != 0) {
        perror("rename checkpoint");
        exit(EXIT_FAILURE);
    }
}


static checkpoint_t load_checkpoint(const char *filename)
{
    checkpoint_t checkpoint;

    FILE *fp = fopen(filename, "rb");

    if (fp == NULL) {
        perror("fopen checkpoint");
        exit(EXIT_FAILURE);
    }

    if (fread(
            &checkpoint,
            sizeof(checkpoint),
            1,
            fp
        ) != 1) {

        fprintf(stderr, "Failed to read checkpoint.\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    fclose(fp);

    if (!checkpoint_is_valid(&checkpoint)) {
        fprintf(stderr, "Invalid checkpoint.\n");
        exit(EXIT_FAILURE);
    }

    return checkpoint;
}


/* ------------------------------------------------------------------ */
/* CLI                                                                 */
/* ------------------------------------------------------------------ */

static uint64_t parse_uint64(
    const char *text,
    const char *name
)
{
    char *end = NULL;

    errno = 0;

    unsigned long long value =
        strtoull(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "Invalid %s: %s\n", name, text);
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

    double value = strtod(text, &end);

    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "Invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }

    return value;
}


static void usage(const char *program)
{
    printf(
        "Usage: %s [OPTIONS]\n"
        "  --darts N\n"
        "  --seed N\n"
        "  --report-every N\n"
        "  --checkpoint-every N\n"
        "  --checkpoint-file FILE\n"
        "  --sleep SEC\n"
        "  --resume\n",
        program
    );
}


static options_t parse_args(int argc, char **argv)
{
    options_t options = {
        .darts = UINT64_C(1000000),
        .seed = UINT64_C(42),
        .report_every = UINT64_C(100000),
        .checkpoint_every = UINT64_C(100000),
        .sleep_seconds = 0.0,
        .checkpoint_file = "pi_signal.bin",
        .resume = 0
    };

    enum {
        OPT_SEED = 1000,
        OPT_REPORT,
        OPT_CKPT_EVERY,
        OPT_CKPT_FILE,
        OPT_SLEEP,
        OPT_RESUME
    };

    static const struct option long_options[] = {
        {"darts", required_argument, NULL, 'n'},
        {"seed", required_argument, NULL, OPT_SEED},
        {"report-every", required_argument, NULL, OPT_REPORT},
        {"checkpoint-every", required_argument, NULL, OPT_CKPT_EVERY},
        {"checkpoint-file", required_argument, NULL, OPT_CKPT_FILE},
        {"sleep", required_argument, NULL, OPT_SLEEP},
        {"resume", no_argument, NULL, OPT_RESUME},
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
                options.darts =
                    parse_uint64(optarg, "--darts");
                break;

            case OPT_SEED:
                options.seed =
                    parse_uint64(optarg, "--seed");
                break;

            case OPT_REPORT:
                options.report_every =
                    parse_uint64(optarg, "--report-every");
                break;

            case OPT_CKPT_EVERY:
                options.checkpoint_every =
                    parse_uint64(optarg, "--checkpoint-every");
                break;

            case OPT_CKPT_FILE:
                options.checkpoint_file = optarg;
                break;

            case OPT_SLEEP:
                options.sleep_seconds =
                    parse_double(optarg, "--sleep");
                break;

            case OPT_RESUME:
                options.resume = 1;
                break;

            case 'h':
                usage(argv[0]);
                exit(EXIT_SUCCESS);

            default:
                exit(EXIT_FAILURE);
        }
    }

    if (options.darts == 0 || options.seed == 0) {
        fprintf(stderr, "darts and seed must be nonzero.\n");
        exit(EXIT_FAILURE);
    }

    return options;
}


/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    options_t options = parse_args(argc, argv);

    install_signal_handlers();

    uint64_t completed_darts = 0;
    uint64_t inside_circle = 0;
    uint64_t rng_state = options.seed;

    double previous_elapsed = 0.0;

    if (options.resume) {

        checkpoint_t checkpoint =
            load_checkpoint(options.checkpoint_file);

        if (
            checkpoint.total_darts != options.darts ||
            checkpoint.seed != options.seed
        ) {
            fprintf(
                stderr,
                "Checkpoint configuration does not match this run.\n"
            );
            return EXIT_FAILURE;
        }

        completed_darts = checkpoint.completed_darts;
        inside_circle = checkpoint.inside_circle;
        rng_state = checkpoint.rng_state;
        previous_elapsed = checkpoint.elapsed_seconds;

        printf(
            "Restarting from checkpoint at dart %" PRIu64 "\n",
            completed_darts
        );
    }

    printf("Process ID: %ld\n", (long) getpid());
    fflush(stdout);

    const double start_time = wall_time();

    for (
        uint64_t i = completed_darts + 1;
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

        /*
         * Safe point for signal-triggered checkpointing.
         */
        if (stop_requested) {

            checkpoint_t checkpoint;

            initialize_checkpoint(&checkpoint);

            checkpoint.completed_darts = i;
            checkpoint.inside_circle = inside_circle;
            checkpoint.rng_state = rng_state;
            checkpoint.total_darts = options.darts;
            checkpoint.seed = options.seed;

            checkpoint.elapsed_seconds =
                previous_elapsed +
                wall_time() -
                start_time;

            printf("\n");
            printf("Signal received: %s\n",
                   signal_name(received_signal));
            printf("Checkpointing at dart %" PRIu64 "...\n", i);

            save_checkpoint(
                options.checkpoint_file,
                &checkpoint
            );

            printf(
                "Checkpoint saved: %s\n",
                options.checkpoint_file
            );

            printf("Exiting cleanly.\n");

            return EXIT_SUCCESS;
        }

        if (
            options.report_every > 0 &&
            (
                i % options.report_every == 0 ||
                i == options.darts
            )
        ) {

            double pi_estimate =
                4.0 * (double) inside_circle /
                (double) i;

            printf(
                "darts = %12" PRIu64
                " / %" PRIu64
                "   pi = %.8f"
                "   error = %.3e\n",
                i,
                options.darts,
                pi_estimate,
                fabs(pi_estimate - PI_TRUE)
            );

            fflush(stdout);

            sleep_seconds(options.sleep_seconds);
        }

        if (
            options.checkpoint_every > 0 &&
            i % options.checkpoint_every == 0
        ) {

            checkpoint_t checkpoint;

            initialize_checkpoint(&checkpoint);

            checkpoint.completed_darts = i;
            checkpoint.inside_circle = inside_circle;
            checkpoint.rng_state = rng_state;
            checkpoint.total_darts = options.darts;
            checkpoint.seed = options.seed;

            checkpoint.elapsed_seconds =
                previous_elapsed +
                wall_time() -
                start_time;

            save_checkpoint(
                options.checkpoint_file,
                &checkpoint
            );

            printf(
                "  --> periodic checkpoint at dart %" PRIu64 "\n",
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
        4.0 * (double) inside_circle /
        (double) options.darts;

    printf("\nCalculation complete.\n");
    printf("Estimated pi   : %.10f\n", pi_estimate);
    printf("Reference pi   : %.10f\n", PI_TRUE);
    printf("Elapsed time   : %.2f s\n", total_elapsed);

    return EXIT_SUCCESS;
}

