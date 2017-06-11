# Fast Fourier Burst Accumulation


Compilation:
```bash
make
```

Uses [Boost Compute](https://github.com/boostorg/compute/) (included), [clFFT](https://github.com/clMathLibraries/clFFT) and [Ceres](http://ceres-solver.org/).

Usage:
```bash
mkdir result
ls example/*.jpg | ./fastfba result/%04d.png
```

*Note*: the image sequence given in *example/* was extracted from [here](http://iie.fing.edu.uy/~mdelbra/videoFA/).
