import sys

# Get file name:
datafile=sys.argv[1]

# Create a list from the input file:
with open(datafile) as f:
  my_list = [ int(i) for i in f ]

print("Max value element : ", max(my_list))

