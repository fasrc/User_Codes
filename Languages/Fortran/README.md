### Purpose:

Example codes illustrating using Fortran on the FASRC cluster.

### Contents:

* Example1: Standard serial Lanczos algorithm with re-orthogonalization
* Example2: Computes integer sum from 1 to 100

### Troubleshooting

If you are using the Intel module 2023.2 or higher, you might see `__msan` errors, for example:

```bash
undefined reference to '__msan_warning_with_origin_noreturn'
undefined reference to '__msan_chain_origin'
undefined reference to '__msan_warning_with_origin_noreturn'
```

This check was introduced by Intel in the 2023.2 in the [check uninit](https://www.intel.com/content/www/us/en/docs/fortran-compiler/developer-guide-reference/2023-2/check.html) which is activated by using the compilation options:

- `-check uninit`
- `-check all`, which includes `uninit`

To resolve this issue, you can be remove `-check uninit` flag. If you are using `-check all`, then use more specific flags instead and don't include `uninit`.

### References:

* [Fortran Tutorial (from tutorialspoint.com)](http://www.tutorialspoint.com/fortran)
