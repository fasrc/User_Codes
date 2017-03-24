#include <stdio.h>
#include <mpfr.h>

void mpfrgetstr (mpfr_t r1, char *chr1, int *nc1, mpfr_exp_t *iexp, mpfr_rnd_t *irnd)
{int n1; int n2;
 n1 = *nc1; n2 = *irnd; 
 mpfr_get_str (chr1, iexp, 10, n1, r1, n2);
}

void mpfrsetstr (mpfr_t r1, char *chr1, mpfr_rnd_t *irnd)
{int n1;
 n1 = *irnd;
 mpfr_set_str (r1, chr1, 10, n1);
}

void mpfroutstr (mpfr_t r1, int *nd)
{int n1; n1 = *nd;
 mpfr_out_str (stdout, 10, n1, r1, MPFR_RNDN); putchar ('\n');
}

void mpfrsetd (mpfr_t r1, double *d, mpfr_rnd_t *irnd)
{double d1; int n1;
 d1 = *d; n1 = *irnd;
 mpfr_set_d (r1, d1, n1);
}

void mpfraddd (mpfr_t r2, mpfr_t r1, double *d, mpfr_rnd_t *irnd)
{double d1; int n1;
 d1 = *d; n1 = *irnd;
 mpfr_add_d (r2, r1, d1, n1);
}

void mpfrdsub (mpfr_t r2, double *d, mpfr_t r1, mpfr_rnd_t *irnd)
{double d1; int n1;
 d1 = *d; n1 = *irnd;
 mpfr_d_sub (r2, d1, r1, n1);
}

void mpfrsubd (mpfr_t r2, mpfr_t r1, double *d, mpfr_rnd_t *irnd)
{double d1; int n1;
 d1 = *d; n1 = *irnd;
 mpfr_sub_d (r2, r1, d1, n1);
}

void mpfrmuld (mpfr_t r2, mpfr_t r1, double *d, mpfr_rnd_t *irnd)
{double d1; int n1;
 d1 = *d; n1 = *irnd;
 mpfr_mul_d (r2, r1, d1, n1);
}

void mpfrddiv (mpfr_t r2, double *d, mpfr_t r1, mpfr_rnd_t *irnd)
{double d1; int n1;
 d1 = *d; n1 = *irnd;
 mpfr_d_div (r2, d1, r1, n1);
}

void mpfrdivd (mpfr_t r2, mpfr_t r1, double *d, mpfr_rnd_t *irnd)
{double d1; int n1;
 d1 = *d; n1 = *irnd;
 mpfr_div_d (r2, r1, d1, n1);
}

void mpfrroot (mpfr_t r2, mpfr_t r1, int *nrt, mpfr_rnd_t *irnd)
{unsigned long int n1; int n2;
 n1 = *nrt; n2 = *irnd;
 mpfr_root (r2, r1, n1, n2);
}

void mpfrpowsi (mpfr_t r2, mpfr_t r1, int *iexp, mpfr_rnd_t *irnd)
{long int n1; int n2;
 n1 = *iexp; n2 = *irnd;
 mpfr_pow_si (r2, r1, n1, n2);
}

void mpfrbesseljn (mpfr_t r2, int *iexp, mpfr_t r1, mpfr_rnd_t *irnd)
{long int n1; int n2;
 n1 = *iexp; n2 = *irnd;
 mpfr_jn (r2, n1, r1, n2);
}

void mpfrbesselyn (mpfr_t r2, int *iexp, mpfr_t r1, mpfr_rnd_t *irnd)
{long int n1; int n2;
 n1 = *iexp; n2 = *irnd;
 mpfr_yn (r2, n1, r1, n2);
}
