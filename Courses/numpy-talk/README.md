# System requirements

Python >= 3.9 ideally with `conda`.

# Building

Ubuntu:
```bash
sudo apt-get install build-essential libomp-dev -y
pip install -r setup-requirements.txt
pip install .
```

Docker:
```bash
# Build
docker build --platform linux/amd64 . -f docker/Dockerfile -t kuramoto
```

# Running

Profile time:
```bash
./bin/main -N 100 -K 2 -T 1000 -p
```

Profile memory:
```bash
mprof run ./bin/main -N 100 -K 2 -T 1000
mprof plot --flame
```

Docker:
```bash
docker run -p 8888:8888 kuramoto 
```

Then navigate to [localhost:8888](localhost:8888)

# References

* https://github.com/xflash96/pybind11_package_example/blob/main/tutorial.md
* https://gameprogrammingpatterns.com/data-locality.html
* https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#vectorizing-functions