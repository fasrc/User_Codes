#include <stdio.h>
#include <gsl/gsl_cblas.h>

int main()
{
    const int n = 5;
    const float alpha = 2.0;
    float x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float y[] = {2.0, 4.0, 6.0, 8.0, 10.0};

    // Perform SAXPY operation
    cblas_saxpy(n, alpha, x, 1, y, 1);

    // Print final values
    printf("SAXPY result: ");
    for (int i = 0; i < n; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");

    return 0;
}
