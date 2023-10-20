# Fortran

![Fortran Logo](Images/fortran-logo.png)

[Fortran](https://en.wikipedia.org/wiki/Fortran), short for Formula Translation, is one of the oldest high-level programming languages, first developed by IBM in the 1950s. It was primarily designed for numerical and scientific computing tasks, making it highly suitable for complex mathematical calculations. Known for its efficiency and speed, Fortran has undergone several revisions over the years, with the latest version being Fortran 2018. Despite its age, Fortran remains relevant in fields such as engineering, physics, and research where performance and numerical accuracy are paramount. Its robustness in handling large-scale scientific and engineering applications, array-oriented programming, and its ability to optimize code for parallel computing have contributed to its longevity in the realm of technical computing.

## Examples

* [Example1](Example1/): Standard serial Lanczos algorithm with re-orthogonalization
* [Example2](Example2/): Computes integer sum from 1 to 100

## Troubleshooting

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

## References

* [Fortran Programming Language](https://fortran-lang.org/)
* [Fortran Tutorial (from tutorialspoint.com)](http://www.tutorialspoint.com/fortran)
